import torch
from torch.utils.data import DataLoader
from diffusionDataset import ImageDataset
from VAE import VAE
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

def check_input_range(dataset, batch_size=1):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    global_min, global_max = float("inf"), float("-inf")
    global_mean, global_std = 0, 0
    count = 0

    for i, (img, _) in enumerate(dl):  # DSImg, truth, mask
        # img: shape (B, C, H, W)
        batch_min = img.min().item()
        batch_max = img.max().item()
        batch_mean = img.mean().item()
        batch_std = img.std().item()

        global_min = min(global_min, batch_min)
        global_max = max(global_max, batch_max)
        global_mean += batch_mean
        global_std += batch_std
        count += 1

    global_mean /= count
    global_std /= count

    print(f"Training MRI input stats:")
    print(f"  Value range: [{global_min:.4f}, {global_max:.4f}]")
    print(f"  Mean: {global_mean:8f}, Std: {global_std:.8f}")

def VAEMu(vae, loader, device):
    all_mu = []
    with torch.no_grad():
        for DSimg, truth in tqdm(loader):
            truth = truth.to(device) * 2 - 1
            mu, _ = vae.encoder(truth)  # shape (B, C, H, W)
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)  # (N, C, H, W)
    channel_means = all_mu.mean(dim=[0, 2, 3])
    channel_stds = all_mu.std(dim=[0, 2, 3])
    print(all_mu.shape)
    print(f'Per-channel mean values: {channel_means}')
    print(f'Per-channel std values: {channel_stds}')

def normedLatent(vae, loader, device, mean_tol=2.0, std_tol=2.0):
    latentD, Min, Max = [], 9999999, -1
    latentU, MinU, MaxU = [], 9999999, -1
    latentE, MinE, MaxE = [], 9999999, -1
    meansE, stdsE = [], []
    klElems = []
    vae.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            DSImg, truth = batch[0], batch[1]
            truth = (truth.to(device) * 2.0 - 1.0)  # [-1,1]

            mu, logvar = vae.encoder(truth)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            latents = z

            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,C,H,W)
            klElems.append(kl_elem)

            latentE.append(z.cpu())
            MinE = min(MinE, z.min().item())
            MaxE = max(MaxE, z.max().item())
            meansE.append(z.mean().item())
            stdsE.append(z.std().item())

            B = latents.size(0)
            sample_means = latents.mean(dim=[1, 2, 3])  # (B,)
            sample_stds = latents.std(dim=[1, 2, 3])    # (B,)
            # Before entering to UNet
            latentU.append(latents.cpu())
            MinU = min(MinU, latents.min().item())
            MaxU = max(MaxU, latents.max().item())

            latentD.append(latents.cpu())
            Min = min(Min, latents.min().item())
            Max = max(Max, latents.max().item())

            for b in range(B):
                m, s = sample_means[b].item(), sample_stds[b].item()

                if abs(m) > mean_tol or abs(s - 1.0) > std_tol:
                    print(f"[WARN] Batch {i}, Sample {b}: "
                          f"mean={m:.3f}, std={s:.3f}")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.arange(1, len(meansE) + 1), np.array(meansE))
    axs[1].plot(np.arange(1, len(stdsE) + 1), np.array(stdsE))
    axs[0].set_title('Mean per Sample')
    axs[1].set_title('Std per Sample')
    plt.show()
    plt.close()
    latentE = torch.cat(latentE, dim=0)
    latentU = torch.cat(latentU, dim=0)
    latentD = torch.cat(latentD, dim=0)
    klElems = torch.cat(klElems, dim=0)
    print(f"After VAE's encoder, mean: {latentE.mean():.2f}, std: {latentE.std():.2f}, max: {MaxE:.2f}, min: {MinE:.2f}")
    print(f"KL per channel: {klElems.mean(dim=(0, 2, 3))}")
    print(f"Before entering to UNet, mean: {latentU.mean():.2f}, std: {latentU.std():.2f}, max: {MaxU:.2f}, min: {MinU:.2f}")
    print(f"Before entering to VAE's decoder, mean: {latentD.mean():.2f}, std: {latentD.std():.2f}, max: {Max:.2f}, min: {Min:.2f}")
    print("✓ Latent inspection done.")


if __name__ == "__main__":
    vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    dataset = ImageDataset('train', device=device)
    check_input_range(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    VAEMu(vae, loader, device)
    normedLatent(vae, loader, device)