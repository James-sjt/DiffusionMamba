import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models
import lpips

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()
for p in lpips_fn.parameters(): p.requires_grad_(False)

def perceptual_loss_lpips_per_sample(pred, target):
    pred_in = pred
    target_in = target
    if pred.min() >= 0 and pred.max() <= 1:
        pred_in = pred * 2.0 - 1.0
        target_in = target * 2.0 - 1.0

    if pred_in.shape[1] == 1:
        pred_in = pred_in.repeat(1, 3, 1, 1)
        target_in = target_in.repeat(1, 3, 1, 1)

    with torch.no_grad():
        tgt = target_in.detach()
    val = lpips_fn.forward(pred_in, tgt)
    return val.view(val.shape[0])  # (B,)

def fft_loss_per_sample(
    pred, target,
    use_log=True,
    highfreq_weight=True,
    normalize_spectrum=True,
    eps=1e-6,
    freq_power=1.5
):
    orig_dtype = pred.dtype
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)

    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)

    pred_f = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
    targ_f = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

    pred_mag = torch.abs(pred_f)
    targ_mag = torch.abs(targ_f)

    if normalize_spectrum:
        def _norm(x):
            denom = x.sum(dim=(-2, -1), keepdim=True).abs() + eps
            return x / denom
        pred_mag = _norm(pred_mag)
        targ_mag = _norm(targ_mag)

    if use_log:
        pred_mag = torch.log(pred_mag + eps)
        targ_mag = torch.log(targ_mag + eps)

    diff = torch.abs(pred_mag - targ_mag)  # (B,C,H,Wf)

    if highfreq_weight:
        B, C, H, Wf = diff.shape
        fy = torch.fft.fftfreq(pred.shape[-2], device=pred.device, dtype=torch.float32)[:, None]  # (H,1)
        fx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device, dtype=torch.float32)[None, :]   # (1,Wf)
        freq = torch.sqrt(fy ** 2 + fx ** 2)  # (H,Wf)
        freq = freq / (freq.max() + 1e-12)
        if freq_power != 1.0:
            freq = freq ** freq_power
        weight = (0.1 + 0.9 * freq)[None, None, :, :]  # (1,1,H,Wf)
        diff = diff * weight

    per_sample = diff.view(diff.shape[0], -1).mean(dim=1)  # (B,)

    return per_sample.to(orig_dtype)


class LaplacianLoss(nn.Module):
    def __init__(self, channels=None, dtype=torch.float32):
        super().__init__()
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x, y, per_sample=False):
        kernel = self.kernel.to(dtype=x.dtype, device=x.device)

        C = x.size(1)
        k = kernel.expand(C, 1, 3, 3)
        lap_x = F.conv2d(x, k, padding=1, groups=C)
        lap_y = F.conv2d(y, k, padding=1, groups=C)

        if per_sample:
            per = (lap_x - lap_y).abs().view(lap_x.shape[0], -1).mean(dim=1)  # (B,)
            return per
        else:
            return F.l1_loss(lap_x, lap_y)

class PatchPerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 4, 7], patch_size=None, device="cuda"):
        super().__init__()
        self.device = device
        self.backbone = models.mobilenet_v2(pretrained=True).features.eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layers = layers
        self.patch_size = patch_size

    def extract_features(self, x):
        feats = []
        out = x
        for idx, layer in enumerate(self.backbone):
            out = layer(out)
            if idx in self.layers:
                feats.append(out)
        return feats

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        device = pred.device

        if self.patch_size is not None:
            ph, pw = self.patch_size, self.patch_size
            pred = pred.unfold(2, ph, ph).unfold(3, pw, pw)
            target = target.unfold(2, ph, ph).unfold(3, pw, pw)
            pred = pred.contiguous().view(-1, C, ph, pw)
            target = target.contiguous().view(-1, C, ph, pw)

        pred_feats = self.extract_features(pred)
        target_feats = self.extract_features(target)

        # L1 loss on features
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.l1_loss(pf, tf)
        return loss

def contrast_loss(pred, target):
    pred_std = pred.flatten(1).std(dim=1)
    tgt_std = target.flatten(1).std(dim=1)
    return F.l1_loss(pred_std, tgt_std * 1.2)

def local_contrast_loss(pred, target, patch=8):
    B, C, H, W = pred.shape
    pred_patches = rearrange(pred, 'b c (h ph) (w pw) -> b c (h w) ph pw', ph=patch, pw=patch)
    tgt_patches  = rearrange(target, 'b c (h ph) (w pw) -> b c (h w) ph pw', ph=patch, pw=patch)
    pred_std = pred_patches.flatten(-2).std(dim=-1)
    tgt_std  = tgt_patches.flatten(-2).std(dim=-1)
    return F.l1_loss(pred_std, tgt_std)