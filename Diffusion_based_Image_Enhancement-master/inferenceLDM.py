from diffusionDataset import ImageDataset
from VAE import VAE
import torch
from latentDiffusion import TinyUNet, sample_img2img_latent_ddpm, save_tensor_as_tif
import os
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

unet_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="unet.pth")
vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")

vae = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)

unet = TinyUNet(in_ch=16, base_ch=64).to(device)

state_dict = torch.load(unet_path, map_location=device, weights_only=True)
unet.load_state_dict(state_dict)
unet.eval()
for p in unet.parameters():
    p.requires_grad_(False)

dataset_valid = ImageDataset('valid', device, True)
dl_val = DataLoader(dataset_valid, batch_size=4, shuffle=False)
os.makedirs("./ldmSample/inputImg", exist_ok=True)
os.makedirs("./ldmSample/denoisedImg", exist_ok=True)
os.makedirs("./ldmSample/truth", exist_ok=True)

# pick first sample
for idx, (img, _, mask) in enumerate(dl_val):  # img: image with noise  _: image without noise
    recon = sample_img2img_latent_ddpm(vae, unet, img, T=1000, start_t=50, device=device)
    # recon: (B, C_img, H, W)
    b = recon.shape[0]
    for i in range(b):
        fname = str(i).zfill(4)
        save_tensor_as_tif(img[i].squeeze(0).cpu(), f"./ldmSample/inputImg/img_{fname}.tif")
        save_tensor_as_tif(recon[i].cpu(), f"./ldmSample/denoisedImg/recon_{fname}.tif")
        if mask is not None:
            save_tensor_as_tif(_[i].squeeze(0).cpu(), f"./ldmSample/truth/truth_{fname}.tif")
    break
