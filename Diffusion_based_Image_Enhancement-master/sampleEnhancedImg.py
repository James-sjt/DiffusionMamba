import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from VAE import VAE
from latentDiffusion import TinyUNet, sample_img2img_latent_ddpm, save_tensor_as_tif
from diffusionDataset import ImageDataset
import os
from huggingface_hub import hf_hub_download

def pathHelper(prefix, idx):
    maskPath = os.path.join(prefix, 'mask', 'mask_' + str(idx).zfill(4) + '.tif')
    enhancedPath = os.path.join(prefix, 'enhancedImg', 'enhanced_' + str(idx).zfill(4) + '.tif')
    imgPath = os.path.join(prefix, 'Img', 'img_' + str(idx).zfill(4) + '.tif')
    return maskPath, enhancedPath, imgPath

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset('valid', device, maskFlag=True)

    dl = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    unet_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="unet.pth")
    vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")

    vae = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()

    unet = TinyUNet(in_ch=16, base_ch=64).to(device)
    state_dict = torch.load(unet_path, map_location=device, weights_only=True)
    unet.load_state_dict(state_dict)
    unet.eval()

    if not os.path.exists("../segMamba/dataSeg"):
        os.mkdir("../segMamba/dataSeg")

    if not os.path.exists("../segMamba/dataSeg/valid/mask"):
        os.mkdir("../segMamba/dataSeg/valid")
        os.mkdir("../segMamba/dataSeg/valid/mask")
        os.mkdir("../segMamba/dataSeg/valid/enhancedImg")
        os.mkdir("../segMamba/dataSeg/valid/Img")
    # sampling valid data
    print("Start Sampling Enhanced Images...")
    startTime = time.time()
    with torch.no_grad():
        idx = 0
        for DSImg, truth, mask in tqdm(dl):
            DSImg, truth = DSImg.to(device), truth.to(device)
            recon = sample_img2img_latent_ddpm(
                vae, unet, DSImg, T=1000, start_t=50, device=device
            )
            B = recon.shape[0]
            for i in range(B):
                enhancedImage = recon[i].squeeze().cpu()
                img = DSImg[i].squeeze().cpu()
                maskTemp = mask[i].squeeze().cpu()

                maskPath, enhancedPath, imgPath = pathHelper('../segMamba/dataSeg/valid', idx)

                save_tensor_as_tif(enhancedImage, enhancedPath)
                save_tensor_as_tif(img, imgPath)
                save_tensor_as_tif(maskTemp, maskPath)

                idx += 1

    # sampling training data
    if not os.path.exists("../segMamba/dataSeg/train/mask"):
        os.mkdir("../segMamba/dataSeg/train")
        os.mkdir("../segMamba/dataSeg/train/mask")
        os.mkdir("../segMamba/dataSeg/train/enhancedImg")
        os.mkdir("../segMamba/dataSeg/train/Img")

    dataset = ImageDataset('train', device, maskFlag=True)
    dl = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    with torch.no_grad():
        idx = 0
        for DSImg, truth, mask in tqdm(dl):
            DSImg, truth = DSImg.to(device), truth.to(device)
            recon = sample_img2img_latent_ddpm(
                vae, unet, DSImg, T=1000, start_t=50, device=device
            )
            B = recon.shape[0]
            for i in range(B):
                enhancedImage = recon[i].squeeze().cpu()
                img = DSImg[i].squeeze().cpu()
                maskTemp = mask[i].squeeze().cpu()

                maskPath, enhancedPath, imgPath = pathHelper('../segMamba/dataSeg/train', idx)

                save_tensor_as_tif(enhancedImage, enhancedPath)
                save_tensor_as_tif(img, imgPath)
                save_tensor_as_tif(maskTemp, maskPath)

                idx += 1
    print(f"Sampling over, time cost: {time.time()-startTime}")


