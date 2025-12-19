import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import copy
import csv
import torchvision.models as models
import time
import numpy as np
from lossFunction import perceptual_loss_lpips_per_sample, fft_loss_per_sample, LaplacianLoss, PatchPerceptualLoss, \
    contrast_loss, local_contrast_loss
import piq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def default_init(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1, 1):
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlockT(nn.Module):
    def __init__(self, in_ch, out_ch=None, time_dim=256, groups=8, use_conv_shortcut=False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )

        if in_ch != out_ch:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            else:
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(groups, in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1)
                )
        else:
            self.shortcut = nn.Identity()

        default_init(self)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        t_out = self.time_mlp(t_emb)
        scale, shift = t_out.chunk(2, dim=1)  # each (B, out_ch)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act2(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, ffn_dim=2048):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ffn_dim, channels, kernel_size=1)
        )

        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

        default_init(self)

    def forward(self, x):
        B, C, H, W = x.shape

        x_ln = self.ln1(x.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        qkv = self.qkv(x_ln)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t):
            return t.reshape(B, self.num_heads, C // self.num_heads, H * W)

        qh = reshape_heads(q)
        kh = reshape_heads(k)
        vh = reshape_heads(v)

        qh = qh / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bhnc,bhmc->bhnm", qh, kh)  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhmc->bhnc", attn, vh)  # (B, heads, C//heads, N)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        out = self.ln2(out.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        out = self.ffn(out)

        return out + x


class MidBlockWithCrossAttn(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, num_heads=16, depth=4):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels, time_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "res": ResBlockT(out_channels, out_channels, time_dim)
            }))

        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, t_emb):
        h = self.in_res(x, t_emb)

        for layer in self.layers:
            h = layer["self_attn"](h)

            h = layer["res"](h, t_emb)

        return h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, num_heads=8, depth=1):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels, time_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "res": ResBlockT(out_channels, out_channels, time_dim)
            }))
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, t_emb):
        h = self.in_res(x, t_emb)

        for layer in self.layers:
            h = layer["self_attn"](h)

            h = layer["res"](h, t_emb)

        return self.pool(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, num_heads=8, depth=1):
        super().__init__()
        self.depth = depth
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Nearest-neighbor upsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Learnable refinement
        )
        self.in_res = ResBlockT(in_channels, out_channels, time_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "res": ResBlockT(out_channels, out_channels, time_dim)
            }))
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, skip, t_emb):
        x = self.up(x)

        x = torch.cat([x, skip], dim=1)
        h = self.in_res(x, t_emb)  # expand channels

        for layer in self.layers:
            h = layer["self_attn"](h)
            h = layer["res"](h, t_emb)
        return h


class TinyUNet(nn.Module):
    def __init__(self, in_ch=16, base_ch=64, time_dim=256):  # cond_dim=512
        super().__init__()
        self.time_dim = time_dim

        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # input conv
        self.inc = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # down path
        self.down1 = DownBlock(base_ch, base_ch, time_dim, num_heads=2, depth=2)  # 64
        self.down2 = DownBlock(base_ch, base_ch * 2, time_dim, num_heads=4)  # 128
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_dim, num_heads=8)  # 256

        # middle with cross-attn
        self.mid = MidBlockWithCrossAttn(base_ch * 4, base_ch * 8, time_dim, num_heads=16)  # 512

        # up path
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, time_dim, num_heads=8)  # 256
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, time_dim, num_heads=4)  # 128
        self.up1 = UpBlock(base_ch * 2, base_ch, time_dim, num_heads=2, depth=2)  # 64

        # output
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        # Input conv
        default_init(self.inc)

        # Time MLP
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Down / Up / Mid
        for block in [self.down1, self.down2, self.down3, self.mid, self.up3, self.up2, self.up1]:
            for sub in block.modules():
                default_init(sub)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t, t):
        # prepare time embedding
        temb = sinusoidal_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)

        h0 = self.inc(x_t)  # (B, base_ch, H, W)
        x1, skip1 = self.down1(h0, temb)  # x1: base_ch/ pooled
        x2, skip2 = self.down2(x1, temb)  # x2: base_ch*2
        x3, skip3 = self.down3(x2, temb)  # x3: base_ch*4, h/8, w/8

        m = self.mid(x3, temb)  # (B, base_ch*4, H/8, W/8)

        u3 = self.up3(m, skip3, temb)  # (B, base_ch*2, H/4, W/4)
        u2 = self.up2(u3, skip2, temb)  # (B, base_ch, H/2, W/2)
        u1 = self.up1(u2, skip1, temb)  # (B, base_ch, H, W)

        out = self.out_conv(u1)
        return out


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, schedule="linear"):
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        target_final = 1e-4
        scale = (target_final / alphas_cumprod[-1]).sqrt()
        betas = 1 - (1 - betas) ** scale
        return betas

    elif schedule == "cosine":
        t = np.linspace(0, T, T + 1)
        alphas_cumprod = np.cos((t / T + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.01)
        return torch.tensor(betas, dtype=torch.float32)

    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")


class DiffusionNoiseSchedule:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device
        betas = make_beta_schedule(T, schedule='linear').to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_start + sqrt_om_ac * noise


def p_sample(model, x_t, t, schedule, cond_img=None):
    betas_t = schedule.betas[t].view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = schedule.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    # predict epsilon
    if cond_img is None:
        eps_pred = model(x_t, t)
    else:
        eps_pred = model(x_t, cond_img, t)

    mean = sqrt_recip_alphas_t * (
            x_t - betas_t / (sqrt_one_minus_alphas_cumprod_t + 1e-12) * eps_pred
    )

    posterior_var = schedule.posterior_variance[t].view(-1, 1, 1, 1)

    noise = torch.randn_like(x_t) if (t[0] > 0) else torch.zeros_like(x_t)

    x_prev = mean + torch.sqrt(posterior_var) * noise
    return x_prev, mean


def cosine_ramp(start_w, end_w, current_epoch, start_epoch, end_epoch):
    if current_epoch < start_epoch:
        return start_w
    if current_epoch > end_epoch:
        return end_w
    t = (current_epoch - start_epoch) / (end_epoch - start_epoch)
    return start_w + 0.5 * (1 - math.cos(math.pi * t)) * (end_w - start_w)


def get_loss_weights(epoch):
    ddpm = 1.0
    x0_latent = 0.2
    recon_loss = 0.5

    # perc_loss = cosine_ramp(1.0, 2.0, epoch, 0, 10)
    # fft_loss = cosine_ramp(2.0, 3.0, epoch, 0, 10)
    # lap_loss = cosine_ramp(1.5, 2.0, epoch, 0, 10)
    # patch_perc_loss = cosine_ramp(0.0, 0.15, epoch, 10, 25)
    # contrast_loss_w = cosine_ramp(1.0, 2.0, epoch, 10, 25)
    # local_contrast_loss_w = cosine_ramp(0.3, 1.2, epoch, 10, 25)

    perc_loss = 1.0
    fft_loss = 2.0
    lap_loss = 1.5
    patch_perc_loss = 0.15
    contrast_loss_w = 6.6
    local_contrast_loss_w = 2.0

    return {
        "ddpm": ddpm,  # matching noise
        "x0_latent": x0_latent,  # L1 loss between x0_latent and x0_latent_bar
        "recon_loss": recon_loss,  # L1 loss between real images and generated images
        "perc_loss": perc_loss,  # perceptual features difference between real images and generated images, VGG16 based
        "fft_loss": fft_loss,  # frequency domain consistency
        "lap_loss": lap_loss,  # laplacian pyramid difference
        "patch_perc_loss": patch_perc_loss,  # patch-wise perceptual loss mobileNet based
        "contrast_loss": contrast_loss_w,  # image-wise standard deviation between prediction and target
        "local_contrast_loss": local_contrast_loss_w,
        # aligning mean intensity and contrast (std) between predicted and ground-truth images
    }


def train_latent_ddpm(
        vae,
        unet,
        dataset,
        T=1000,
        device="cuda",
        epochs=100,
        batch_size=8,
        lr=1e-3,
        ckpt_dir="./ckpts_unet",
        use_ema=True,
        ema_beta=0.999,
        use_aux_recon_loss=True,
        grad_clip=1.0,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    schedule = DiffusionNoiseSchedule(T=T, device=device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)
    batches_per_epoch = len(dataloader)
    total_steps = epochs * batches_per_epoch
    warmup_steps = 5000
    log_file_U = "training_log_UNet.csv"
    if not os.path.exists(log_file_U):
        with open(log_file_U, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "avg_loss", "avg_ddpm", "avg_x0_latent", "avg_recon_aux", "avg_perc", "avg_fft", 'avg_lap',
                 'avg_patch_perc', "avg_contrast", "avg_local_contrast", "lr"])
    def warmup_lambda(step):
        return min(1.0, step / warmup_steps)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_steps),
        eta_min=5e-5
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    mse = nn.MSELoss(reduction='mean')
    l1 = nn.L1Loss(reduction='none')

    if use_ema:
        ema_unet = copy.deepcopy(unet).eval()
        ema_unet.to(device)
        for p in ema_unet.parameters(): p.requires_grad_(False)
    else:
        ema_unet = None

    bestPSNR = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        loss_weights = get_loss_weights(epoch + 1)
        epoch_loss = 0
        total_x0_latent = 0
        total_recon_aux = 0
        total_perc = 0
        total_fft = 0
        total_ddpm = 0
        total_lap = 0
        total_patch_perc = 0
        total_contrast = 0
        total_local_contrast = 0

        for batch in pbar:
            DSImg, truth = batch[0], batch[1]
            truth = (truth.to(device) * 2.0 - 1.0)  # [-1,1]
            DSImg = (DSImg.to(device) * 2.0 - 1.0)  # [-1,1]

            with torch.no_grad():
                mu, logvar = vae.encoder(DSImg)  # (B,C,H,W)
                latents = vae.reparameterize(mu, logvar).detach()

            B = latents.shape[0]
            t = torch.randint(0, T, (B,), device=device).long()
            noise = torch.randn_like(latents, device=device)
            # forward q(z_t|z0)
            x_t = schedule.q_sample(latents, t, noise=noise)

            eps_pred = unet(x_t, t)
            loss_ddpm = mse(eps_pred, noise) * loss_weights["ddpm"]
            total_ddpm += loss_ddpm.item()

            if use_aux_recon_loss:
                sqrt_ac = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_om_ac = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

                x0_pred = (x_t - sqrt_om_ac * eps_pred) / (sqrt_ac + 1e-12)  # (B,C,H,W)
                loss_x0_latent = F.smooth_l1_loss(x0_pred, latents, reduction='mean') * loss_weights['x0_latent']

                recon_img = vae.decoder(x0_pred)

                truth_img = truth  # already [-1,1]
                x0_pred_img01 = (recon_img + 1) / 2  # b, c, 256, 256
                truth_img01 = (truth_img + 1) / 2

                if x0_pred_img01.size(1) == 1:
                    x0_pred_img013 = x0_pred_img01.repeat(1, 3, 1, 1)
                    truth_img013 = truth_img01.repeat(1, 3, 1, 1)

                lap_loss_fn = LaplacianLoss().to(device)
                patch_loss_fn = PatchPerceptualLoss(patch_size=16).to(device)
                loss_lap = lap_loss_fn(x0_pred_img01, truth_img01) * loss_weights['lap_loss']
                loss_patch_perc = patch_loss_fn(x0_pred_img013, truth_img013) * loss_weights['patch_perc_loss']

                # L1 per-sample (image domain).
                l1_per_px = l1(x0_pred_img01, truth_img01)  # (B,1,H,W)
                l1_per_sample = l1_per_px.view(B, -1).mean(dim=1)  # (B,)
                loss_recon_aux = l1_per_sample.mean() * loss_weights['recon_loss']

                # LPIPS per-sample
                lpips_per_sample = perceptual_loss_lpips_per_sample(x0_pred_img013, truth_img013)  # pass in [0,1]
                loss_perc = lpips_per_sample.mean() * loss_weights['perc_loss']

                # FFT per-sample
                fft_per_sample = fft_loss_per_sample(x0_pred_img01, truth_img01)
                loss_fft = fft_per_sample.mean() * loss_weights['fft_loss']

                # contrast loss
                loss_contrast = contrast_loss(x0_pred_img01, truth_img01) * loss_weights['contrast_loss']

                # local contrast loss
                loss_local_contrast = local_contrast_loss(x0_pred_img01, truth_img01) * loss_weights['local_contrast_loss']

            else:
                loss_x0_latent = 0.
                loss_recon_aux = 0.
                loss_perc = 0.
                loss_fft = 0.
                loss_lap = 0.
                loss_patch_perc = 0.
                loss_contrast = 0.
                loss_local_contrast = 0.

            loss = loss_ddpm + loss_x0_latent + loss_recon_aux + loss_perc + loss_fft + loss_lap + loss_patch_perc + loss_contrast + loss_local_contrast

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            if ema_unet is not None:
                update_ema(ema_unet, unet, beta=ema_beta)

            epoch_loss += loss.item()
            total_x0_latent += (loss_x0_latent if isinstance(loss_x0_latent, float) else loss_x0_latent.item())
            total_recon_aux += (loss_recon_aux if isinstance(loss_recon_aux, float) else loss_recon_aux.item())
            total_perc += (loss_perc if isinstance(loss_perc, float) else loss_perc.item())
            total_fft += (loss_fft if isinstance(loss_fft, float) else loss_fft.item())
            total_lap += (loss_lap if isinstance(loss_lap, float) else loss_lap.item())
            total_patch_perc += (loss_patch_perc if isinstance(loss_patch_perc, float) else loss_patch_perc.item())
            total_contrast += (loss_contrast if isinstance(loss_contrast, float) else loss_contrast.item())
            total_local_contrast += (
                loss_local_contrast if isinstance(loss_local_contrast, float) else loss_local_contrast.item())
            pbar.set_postfix(
                loss=loss.item(),
                ddpm=loss_ddpm.item(),
                x0_latent=(loss_x0_latent if isinstance(loss_x0_latent, float) else loss_x0_latent.item()),
                recon_aux=(loss_recon_aux if isinstance(loss_recon_aux, float) else loss_recon_aux.item()),
                perc=(loss_perc if isinstance(loss_perc, float) else loss_perc.item()),
                fft=(loss_fft if isinstance(loss_fft, float) else loss_fft.item()),
                lap=(loss_lap if isinstance(loss_lap, float) else loss_lap.item()),
                patch=(loss_patch_perc if isinstance(loss_patch_perc, float) else loss_patch_perc.item()),
                contrast=(loss_contrast if isinstance(loss_contrast, float) else loss_contrast.item()),
                local_contrast=(
                    loss_local_contrast if isinstance(loss_local_contrast, float) else loss_local_contrast.item()),
            )

        avg_loss = epoch_loss / len(dataloader)
        avg_x0_latent = total_x0_latent / len(dataloader)
        avg_recon_aux = total_recon_aux / len(dataloader)
        avg_perc = total_perc / len(dataloader)
        avg_fft = total_fft / len(dataloader)
        avg_ddpm = total_ddpm / len(dataloader)
        avg_lap = total_lap / len(dataloader)
        avg_patch_perc = total_patch_perc / len(dataloader)
        avg_contrast = total_contrast / len(dataloader)
        avg_local_contrast = total_local_contrast / len(dataloader)
        print(
            f"Stats of UNet on training set: Epoch {epoch + 1}/{epochs} avg loss: {avg_loss:.4f} avg_ddpm: {avg_ddpm:.4f} avg_x0_latent: {avg_x0_latent:.4f} avg_recon_aux: {avg_recon_aux:.4f} avg_perc: {avg_perc:.4f} avg_fft: {avg_fft:.4f} lap: {avg_lap:.4f} patch: {avg_patch_perc:.4f} contrast: {avg_contrast:.4f} local contrast: {avg_local_contrast:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")
        with open(log_file_U, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, avg_ddpm, avg_x0_latent, avg_recon_aux, avg_perc, avg_fft, avg_lap,
                             avg_patch_perc, avg_contrast, avg_local_contrast, scheduler.get_last_lr()[0]])

        if (epoch + 1) % 5 == 0:
            print('**********Validation Start**********')
            starTime = time.time()
            if ema_unet is not None:
                valid_loss = valid(ema_unet, schedule)
            else:
                valid_loss = valid(unet, schedule)
            print(f'Total validation Time: {time.time() - starTime:.2f} seconds')
            if valid_loss[0] > bestPSNR:
                bestPSNR = valid_loss[0]
                save_name = os.path.join(ckpt_dir, f"unet_best_epoch{epoch + 1}.pth")
                if ema_unet is not None:
                    torch.save(ema_unet.state_dict(), save_name)
                else:
                    torch.save(unet.state_dict(), save_name)
                print(f"Saved best UNet -> {save_name}")

            save_name = os.path.join(ckpt_dir, f"unet_checkPoint_epoch{epoch + 1}.pth")
            if ema_unet is not None:
                torch.save(ema_unet.state_dict(), save_name)
            else:
                torch.save(unet.state_dict(), save_name)
            print(f"Saved best UNet -> {save_name}")
            print('**********Validation Over**********')

    # final save
    final_name = os.path.join(ckpt_dir, "unet_final.pth")
    if ema_unet is not None:
        torch.save(ema_unet.state_dict(), final_name)
    else:
        torch.save(unet.state_dict(), final_name)
    print(f"Saved final UNet -> {final_name}")


@torch.no_grad()
def valid(model, schedule, T=50):
    dataset = ImageDataset('valid', device)
    dataloader = DataLoader(dataset, batch_size=8)
    psnr_total, ssim_total, gmsd_total = 0, 0, 0
    for DSImg, truth in dataloader:
        truth = truth.to(device)
        DSImg = (DSImg.to(device) * 2.0 - 1.0)  # [-1,1]

        mu, logvar = vae.encoder(DSImg)  # (B,C,H,W)
        std = torch.exp(0.5 * logvar)
        eps_z = torch.randn_like(std)
        latents = (mu + eps_z * std).detach()

        x_t = latents

        t = torch.full((x_t.shape[0],), T).to(device)

        eps_pred = model(x_t, t)

        sqrt_ac = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om_ac = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x0_pred = (x_t - sqrt_om_ac * eps_pred) / (sqrt_ac + 1e-12)  # (B,C,H,W)
        recon_img = vae.decoder(x0_pred)

        x0_pred_img01 = (recon_img + 1) / 2  # b, c, 256, 256

        psnr_score = piq.psnr(x0_pred_img01, truth)
        psnr_total += psnr_score.item()

        ssim_score = piq.ssim(x0_pred_img01, truth)
        ssim_total += ssim_score.item()

        gmsd_score = piq.gmsd(x0_pred_img01, truth)
        gmsd_total += gmsd_score.item()
    avg_psnr, avg_ssim, avg_gmsd = psnr_total / len(dataloader), ssim_total / len(dataloader), gmsd_total / len(
        dataloader)
    print(f"Validation--- PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, GMSD: {avg_gmsd:.4f}")
    return avg_psnr, avg_ssim, avg_gmsd


@torch.no_grad()
def update_ema(ema_model, model, beta=0.999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(beta).add_(p.data, alpha=1 - beta)


def sample_img2img_latent_ddpm(
        vae,
        unet,
        noisy_image,
        T=1000,
        start_t=50,
        device="cuda",
):
    unet.eval()
    vae.eval()
    noisy_image = (noisy_image * 2 - 1).to(device)  # normalize to [-1, 1]

    with torch.no_grad():
        mu, logvar = vae.encoder(noisy_image)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_noisy = mu + eps * std
        B = z_noisy.size(0)

        schedule = DiffusionNoiseSchedule(T=T, device=device)
        # eps = torch.randn_like(z_noisy, device=device)
        # sqrt_ac = schedule.sqrt_alphas_cumprod[start_t].view(1, 1, 1, 1)
        # sqrt_om_ac = schedule.sqrt_one_minus_alphas_cumprod[start_t].view(1, 1, 1, 1)
        # x_t = sqrt_ac * z_noisy + sqrt_om_ac * eps

        x_prev = z_noisy
        for time in range(start_t, -1, -1):
            t_batch = torch.full((B,), time, device=device, dtype=torch.long)
            x_prev, mean = p_sample(unet, x_prev, t_batch, schedule, cond_img=None)

        recon_final = vae.decoder(x_prev)
        recon_final = (recon_final + 1) / 2  # normalize to [0, 1]
        return recon_final


# ---- Utilities for saving .tif with zfill naming ----
def save_tensor_as_tif(img_tensor, path):
    x = img_tensor.detach().cpu().float()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    pil = to_pil_image(x)
    pil.save(path)


if __name__ == "__main__":
    from diffusionDataset import ImageDataset
    from VAE import VAE
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # instantiate UNet
    unet = TinyUNet().to(device)
    # unet_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="unet.pth")
    # unet.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    unet.train()

    # ---- TRAIN ----
    train_ds = ImageDataset('train', device)
    train_latent_ddpm(
        vae=vae,
        unet=unet,
        dataset=train_ds,
        T=1000,
        device=device,
        epochs=100,
        batch_size=8,
        lr=1e-4,
        ckpt_dir="./ckpts_unet"
    )
