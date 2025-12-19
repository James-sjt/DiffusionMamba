import piq
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from VAE import VAE
from latentDiffusion import TinyUNet, sample_img2img_latent_ddpm
from diffusionDataset import ImageDataset
from huggingface_hub import hf_hub_download


@torch.no_grad()
def evaluate_image_quality(enhanced, ground_truth, lpips_metric=None):
    assert enhanced.shape == ground_truth.shape, "Shape mismatch!"
    assert 0.0 <= enhanced.min() and enhanced.max() <= 1.0, "Enhanced must be in [0,1]"
    assert 0.0 <= ground_truth.min() and ground_truth.max() <= 1.0, "GT must be in [0,1]"

    metrics = {
        "PSNR": piq.psnr(enhanced, ground_truth, data_range=1.0),
        "SSIM": piq.ssim(enhanced, ground_truth, data_range=1.0),
        "GMSD": piq.gmsd(enhanced, ground_truth, data_range=1.0),
    }

    if lpips_metric is not None:
        enhanced_rgb = enhanced.repeat(1, 3, 1, 1)
        ground_truth_rgb = ground_truth.repeat(1, 3, 1, 1)
        metrics["LPIPS"] = lpips_metric(enhanced_rgb, ground_truth_rgb)

    return metrics


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset('valid', device)

    dl = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    vae = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()

    unet = TinyUNet(in_ch=16, base_ch=64).to(device)
    unet_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="unet.pth")
    state_dict = torch.load(unet_path, map_location=device, weights_only=True)
    unet.load_state_dict(state_dict)
    unet.eval()

    lpips_metric = piq.LPIPS(replace_pooling=True, reduction='mean').to(device)

    total = {"PSNR": 0, "SSIM": 0, "GMSD": 0, "LPIPS": 0}
    inference_time = 0.0

    with torch.no_grad():
        for DSImg, truth in tqdm(dl):
            DSImg, truth = DSImg.to(device), truth.to(device)

            start_time = time.time()
            recon = sample_img2img_latent_ddpm(
                vae, unet, DSImg, T=1000, start_t=50, device=device
            )
            inference_time += (time.time() - start_time)

            metrics = evaluate_image_quality(recon, truth, lpips_metric)
            for k in total.keys():
                total[k] += metrics[k].mean().item()

    mean_metrics = {k: v / len(dl) for k, v in total.items()}
    print("Mean metrics:", mean_metrics)
    print("Average inference time per image:", inference_time / len(dl))
