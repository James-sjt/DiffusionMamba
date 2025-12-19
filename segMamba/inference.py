import matplotlib.pyplot as plt
import torch
import os
from segDataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from student import student
from mambaSeg import mambaSeg
from huggingface_hub import hf_hub_download
import argparse

parser = argparse.ArgumentParser(description="Choose a Particular Model to Sample Images: (teacher/student)")
parser.add_argument('--model_name', type=str, default='student')
model_name = parser.parse_args().model_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model_name == 'student':
    model = student().to(device)
    ckpt_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="segModel_studentDistilled.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
elif model_name == 'teacher':
    model = mambaSeg().to(device)
    ckpt_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="segModel_teacher.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)

ds_v = ImageDataset('valid', device, imgType='both')
dl_v = DataLoader(ds_v, batch_size=10, shuffle=True)
model.eval()

if model_name == 'student':
    if not os.path.exists('./studentSamples'):
        os.makedirs('./studentSamples')
elif model_name == 'teacher':
    if not os.path.exists('./teacherSamples'):
        os.makedirs('./teacherSamples')

with torch.no_grad():
    for enhanced, images, masks in tqdm(dl_v):  # batch_size = 5
        images = images  # (B, C, H, W)
        masks = masks     # (B, 1, H, W)

        outputs = model(enhanced, images) # (B, 1, H, W)

        # Binary segmentation: apply sigmoid + threshold
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # Visualize each item in the batch
        batch_size = images.size(0)
        for i in range(batch_size):
            image = images[i].cpu()
            mask = masks[i].cpu().squeeze().numpy()
            pred = preds[i].cpu().squeeze().numpy()

            # If grayscale -> shape (1,H,W) → squeeze channel
            if image.shape[0] == 1:
                image = image.squeeze().numpy()  # (H,W)
                cmap = "gray"
            else:
                image = image.permute(1, 2, 0).numpy()  # (H,W,3)
                cmap = None  # matplotlib auto-handles RGB

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            ax[0].imshow(image, cmap=cmap)
            ax[0].set_title("Input Image")
            ax[0].axis("off")

            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("Ground Truth")
            ax[1].axis("off")

            ax[2].imshow(pred, cmap="gray")
            ax[2].set_title("Prediction")
            ax[2].axis("off")

            if model_name == 'student':
                plt.savefig(f'./studentSamples/sample_{i}.png', bbox_inches='tight', dpi=300)
            elif model_name == 'teacher':
                plt.savefig(f'./teacherSamples/sample_{i}.png', bbox_inches='tight', dpi=300)
            plt.show()

        break  # only first batch

