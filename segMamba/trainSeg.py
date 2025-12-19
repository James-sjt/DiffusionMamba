import torch
from segDataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.optim as optim
from tqdm import tqdm
from lossfunction import Loss
from mambaSeg import mambaSeg
from student import student
import os
import csv
from metric import hd95, nsd, dsc
import time
import gc
import argparse

parser = argparse.ArgumentParser(description="Choose the model you want to train (student/teacher)")
parser.add_argument('--model_name', type=str, default='teacher')
model_name = parser.parse_args().model_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ImageDataset('train', device, imgType="both")
datasetValid = ImageDataset('valid', device, imgType="both")
dl_t = DataLoader(dataset, batch_size=6, shuffle=True)
dl_v = DataLoader(datasetValid, batch_size=4, shuffle=False)
criterion = Loss()
if model_name == 'student':
    model = student().to(device)
elif model_name == 'teacher':
    model = mambaSeg().to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
batches_per_epoch = len(dl_t)
total_steps = 100 * batches_per_epoch
warmup_steps = 5000


def warmup_lambda(step):
    return min(1.0, step / warmup_steps)


warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(total_steps - warmup_steps),
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

if not os.path.exists("./ckpts_unet"):
    os.mkdir("./ckpts_unet")

max_dsc = 0

# CSV setup
csv_filename = "training_loss.csv"
fieldnames = ["epoch", "train_loss", "valid_loss", "train_focal_loss", "train_dice_loss", "train_boundary_loss"]
# Create or open the CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row
    model.train()
    for epoch in range(100):
        total_loss, t_focal, t_dice, t_boundary = 0., 0., 0., 0.
        # Training Loop
        for enhanced, img, mask in tqdm(dl_t):  # enhanced, img, mask
            out = model(img, enhanced)  # enhanced
            loss, focal, dice, boundary = criterion(out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            t_focal += focal.item()
            t_dice += dice.item()
            t_boundary += boundary.item()
            scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

        current_lr = scheduler.get_last_lr()[0]
        mean_focal, mean_dice, mean_boundary = t_focal / len(dl_t), t_dice / len(dl_t), t_boundary / len(dl_t)
        print(
            f"Epoch {epoch + 1}/{100}, Loss of train: {total_loss / len(dl_t):.4f}, Focal: {mean_focal:.4f}, Dice: {mean_dice:.4f}, Boundary: {mean_boundary:.4f}, current lr: {current_lr:.6f}")
        criterion.update_epoch_losses(mean_focal, mean_dice, mean_boundary)

        # Validation Loop
        model.eval()
        total_dsc, total_nsd, total_hd, total_time = 0, 0, 0, 0
        with torch.no_grad():
            for enhanced, img, mask in tqdm(dl_v):
                stat_time = time.time()
                y_pred = model(img, enhanced)
                y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred > 0.5).float()
                total_time += time.time() - stat_time
                total_dsc += dsc(y_pred, mask)
                total_nsd += nsd(y_pred, mask)
                total_hd += hd95(y_pred, mask)
                del enhanced, img, mask, y_pred
            torch.cuda.empty_cache()
            gc.collect()
            avg_dsc = total_dsc / len(dl_v)
            avg_nsd = total_nsd / len(dl_v)
            avg_hd95 = total_hd / len(dl_v)
            avg_time = total_time / len(dl_v)

            print(f'DSC: {avg_dsc:.4f}, NSD: {avg_nsd:.4f}, HD95: {avg_hd95:.4f}, Inference Time: {avg_time:.4f}')
            if avg_dsc > max_dsc:
                if model_name == 'teacher':
                    torch.save(model.state_dict(), f"./segModel_teacher.pth")
                elif model_name == 'student':
                    torch.save(model.state_dict(), f"./segModel_pretrained_student.pth")
                max_dsc = avg_dsc
                print(f"********Epoch: {epoch + 1}, model's parameters saved!********")
        model.train()

        # Write losses to CSV file after each epoch
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": total_loss / len(dl_t),
                "valid_loss": avg_dsc,
                "train_focal_loss": t_focal / len(dl_t),
                "train_dice_loss": t_dice / len(dl_t),
                "train_boundary_loss": t_boundary / len(dl_t)
            })
