import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch
from VAE import VAE
from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")
    model = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    model.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    model.eval()

    img = Image.open("truth.tif").convert("L")   # grayscale
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),        # (C,H,W) in [0,1]
    ])
    x = transform(img).unsqueeze(0).to(device) * 2 - 1  # (1,1,256,256)

    # Inference
    with torch.no_grad():
        recon, mu, logvar = model(x)

    x = ((x.squeeze() + 1) / 2).cpu()  # back to [0, 1]
    recon = ((recon.squeeze() + 1) / 2).cpu()

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Original")
    axs[0].imshow(x, cmap="gray")

    axs[1].set_title("Reconstruction")
    axs[1].imshow(recon, cmap="gray")
    plt.show()

