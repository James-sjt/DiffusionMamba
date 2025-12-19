import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import sys
sys.path.append('../')
from VAE import VAE
from latentDiffusion import save_tensor_as_tif
from GAN_based import Generator
from GANDataset import ImageDataset
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

    gan = Generator().to(device)
    ckpt_path = hf_hub_download(repo_id="James0323/GAN-weights", filename="GAN.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gan.load_state_dict(checkpoint)
    gan.eval()

    if not os.path.exists("./dataSegGAN"):
        os.mkdir("./dataSegGAN")

    if not os.path.exists("./dataSegGAN/valid/mask"):
        os.mkdir("./dataSegGAN/valid")
        os.mkdir("./dataSegGAN/valid/mask")
        os.mkdir("./dataSegGAN/valid/enhancedImg")
        os.mkdir("./dataSegGAN/valid/Img")
    # sampling valid data
    print("Start Sampling Enhanced Images...")
    startTime = time.time()
    with torch.no_grad():
        idx = 0
        for DSImg, truth, mask in tqdm(dl):
            DSImg, truth = DSImg.to(device), truth.to(device)
            recon = (gan(DSImg) + 1) / 2
            B = recon.shape[0]
            for i in range(B):
                enhancedImage = recon[i].squeeze().cpu()
                img = DSImg[i].squeeze().cpu()
                maskTemp = mask[i].squeeze().cpu()

                maskPath, enhancedPath, imgPath = pathHelper('./dataSegGAN/valid', idx)

                save_tensor_as_tif(enhancedImage, enhancedPath)
                save_tensor_as_tif(img, imgPath)
                save_tensor_as_tif(maskTemp, maskPath)

                idx += 1

    # sampling training data
    if not os.path.exists("./dataSegGAN/train/mask"):
        os.mkdir("./dataSegGAN/train")
        os.mkdir("./dataSegGAN/train/mask")
        os.mkdir("./dataSegGAN/train/enhancedImg")
        os.mkdir("./dataSegGAN/train/Img")

    dataset = ImageDataset('train', device, maskFlag=True)
    dl = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    with torch.no_grad():
        idx = 0
        for DSImg, truth, mask in tqdm(dl):
            DSImg, truth = DSImg.to(device), truth.to(device)
            recon = (gan(DSImg) + 1) / 2
            B = recon.shape[0]
            for i in range(B):
                enhancedImage = recon[i].squeeze().cpu()
                img = DSImg[i].squeeze().cpu()
                maskTemp = mask[i].squeeze().cpu()

                maskPath, enhancedPath, imgPath = pathHelper('./dataSegGAN/train', idx)

                save_tensor_as_tif(enhancedImage, enhancedPath)
                save_tensor_as_tif(img, imgPath)
                save_tensor_as_tif(maskTemp, maskPath)

                idx += 1
    print(f"Sampling over, time cost: {time.time()-startTime}")


