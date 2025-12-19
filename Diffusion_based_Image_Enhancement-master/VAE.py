import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusionDataset import ImageDataset
from tqdm import tqdm
import lpips
from pytorch_msssim import ssim

class ResBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, latent_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            ResBlock(64, groups=32),
            ResBlock(64),

            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            ResBlock(128, groups=32),
            ResBlock(128, groups=32),

            # 64 -> 32
            nn.Conv2d(128, hidden_channels, kernel_size=4, stride=2, padding=1),
            ResBlock(hidden_channels, groups=32),
            ResBlock(hidden_channels, groups=32),
            # result: (B, hidden_channels, 32, 32)
        )

        self.mu = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)

    def forward(self, x):
        h = self.encoder(x)         # (B, hidden_channels, 32, 32)
        mu = self.mu(h)             # (B, latent_channels, 32, 32)
        logvar = self.logvar(h)     # (B, latent_channels, 32, 32)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden_channels=128, latent_channels=16):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels, 4, stride=2, padding=1),
            ResBlock(hidden_channels, groups=32),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 64, 4, stride=2, padding=1),
            ResBlock(64, groups=16),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            ResBlock(32, groups=8),
        )
        self.final = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, z):
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        return torch.tanh(self.final(x))


class VAE(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, latent_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_channels)
        self.decoder = Decoder(in_channels, latent_channels=latent_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)

def kl_divergence(mu, logvar, free_bits=0.05, target_kl=0.5, balance_weight=0.5):
    # Compute per-channel KL
    kl_per_channel = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, C, H, W)
    kl_per_channel = kl_per_channel.mean(dim=[0, 2, 3])  # (C,)

    kl_fb = torch.clamp(kl_per_channel, min=free_bits)

    balance_loss = balance_weight * F.mse_loss(
        kl_per_channel, torch.full_like(kl_per_channel, target_kl)
    )

    return kl_fb.mean(), balance_loss

def vae_loss(
    recon_x, x, mu, logvar, perceptual_fn=None,
    recon_weight=1.0, kl_weight=0.01, perceptualLoss=False, ssimLoss=True, edgeLoss=False):
    perceptual_weight = 0.3
    ssim_weight = 0.1
    edge_weight = 0.3

    recon_loss = F.l1_loss(recon_x, x, reduction='mean')

    perceptual_loss = 0
    if perceptualLoss:
        perceptual_loss = perceptual_fn(recon_x, x).mean()

    ssim_loss = 0
    if ssimLoss:
        ssim_val = ssim((recon_x + 1) / 2, (x + 1) / 2, data_range=1.0, size_average=True)  # SSIM ∈ [0,1]
        ssim_loss = 1 - ssim_val

    kl_loss, balance_loss = kl_divergence(mu, logvar, free_bits=0.1, target_kl=0.5, balance_weight=0.1)

    edge_loss = 0
    if edgeLoss:
        edge_loss = gradient_loss(recon_x, x)

    loss = (
            recon_weight * recon_loss +
            (perceptual_weight if perceptualLoss else 0) * perceptual_loss +
            (ssim_weight if ssimLoss else 0) * ssim_loss +
            (edge_weight if edgeLoss else 0) * edge_loss +
            kl_weight * kl_loss + balance_loss
    )

    return loss, recon_loss, perceptual_loss, ssim_loss, kl_loss, edge_loss


def train_one_epoch(model, dataloader, optim, perceptual_fn, recon_weight=1.0, beta=1.0):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(dataloader)):
        DSImg, truth = batch
        DSImg, truth = DSImg.to(device) * 2 - 1, truth.to(device) * 2 - 1  # norm to [-1, 1]
        imgs = torch.cat((DSImg, truth), dim=0).to(device)

        optim.zero_grad()
        recon, mu, logvar = model(imgs)
        loss, recon_l, perceptual_loss, ssim_loss, kl_l, edge_loss = vae_loss(recon, imgs, mu, logvar, perceptual_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-6)

    dataset = ImageDataset('train', device)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    bestLoss = 99999
    perceptual_fn = lpips.LPIPS(net='alex').to(device)
    for epoch in range(1, 101):
        avg_loss = train_one_epoch(model, loader, optimizer, perceptual_fn, recon_weight=1.0, beta=1.0)
        if avg_loss < bestLoss:
            bestLoss = avg_loss
            torch.save(model.state_dict(), f"VAESSIM.pth")
            print(f'********Parameter saved to VAESSIM.pth********')
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{100}, Loss of train: {avg_loss}, current lr: {current_lr:.6f}")
