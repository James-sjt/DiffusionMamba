import matplotlib.pyplot as plt
import torch
import os
from segDataset import ImageDataset
from torch.utils.data import DataLoader
from student import student
from tqdm import tqdm
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = student().to(device)
model.load_state_dict(torch.load('./student/student_epoch32.pth', map_location=device))
ds_v = ImageDataset('valid', device, imgType='enhanced')
dl_v = DataLoader(ds_v, batch_size=10, shuffle=True)
model.eval()

if not os.path.exists('./studentSamples'):
    os.makedirs('./studentSamples')
total_time = 0
with torch.no_grad():
    for images, masks in tqdm(dl_v):  # batch_size = 5
        images = images.to(device)   # (B, C, H, W)
        masks = masks.to(device)     # (B, 1, H, W)
        star_time = time()
        outputs = model(images) # (B, 1, H, W)
        total_time += time() - star_time

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

            plt.savefig(f'./studentSamples/sample_{i}.png', bbox_inches='tight', dpi=300)
            plt.show()

        break  # only first batch
    print(f'Avr inference time: {total_time/batch_size} s')

