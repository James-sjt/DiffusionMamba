import torch
import sys
sys.path.append("../")
from VAE import VAE
from GAN_based import Generator
from GANDataset import ImageDataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate generator architecture
generator = Generator().to(device)

# Load weights
ckpt_path = hf_hub_download(repo_id="James0323/GAN-weights", filename="GAN.pth")
checkpoint = torch.load(ckpt_path, map_location=device)
generator.load_state_dict(checkpoint)

generator.eval()
save_dir = "image_GAN"
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    valid_ds = ImageDataset('valid', device)
    dl_v = DataLoader(valid_ds, batch_size=1, shuffle=False, drop_last=True)
    for i, (DSImg, truth) in enumerate(dl_v):
        DSImg = DSImg.to(device)
        
        fake = (generator(DSImg) + 1) / 2
        vutils.save_image(
            fake,
            os.path.join(save_dir, f"gan_{i:04d}.png"),
            normalize=False
        )

