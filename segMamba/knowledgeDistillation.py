from mambaSeg import mambaSeg
from student import student
from segDataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import torch.optim as optim
from distillationLoss import DistillationLoss
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import csv
import gc
from metric import hd95, nsd, dsc
import time
from huggingface_hub import hf_hub_download

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

teacher = mambaSeg().to(device)
ckpt_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="segModel_teacher.pth")
checkpoint = torch.load(ckpt_path, map_location=device)
teacher.load_state_dict(checkpoint)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

ds_t = ImageDataset('train', device, 'both')
dl_t = DataLoader(ds_t, batch_size=6, shuffle=True)
ds_v = ImageDataset('valid', device, 'both')
dl_v = DataLoader(ds_v, batch_size=4, shuffle=False)

student = student().to(device)
ckpt_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="segModel_pretrained_student.pth")
checkpoint = torch.load(ckpt_path, map_location=device)
student.load_state_dict(checkpoint)
student.train()

optimizer = optim.AdamW(student.parameters(), lr=5e-4) # 1e-4

max_dsc = 0

criterion = DistillationLoss()

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

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# CSV Setup
csv_filename = "student_loss.csv"
fieldnames = ["epoch", "train_loss", "train_soft_loss", "valid_dsc", "valid_nsd", "valid_hd95"]

# Create or open the CSV file and write headers
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

for epoch in range(100):
    train_loss, train_soft_loss, t_focal, t_dice, t_boundary = 0, 0, 0, 0, 0
    for enhanced, img, mask in tqdm(dl_t):
        studentOut = student(img, enhanced)
        with torch.no_grad():
            teacherOut = teacher(img, enhanced)
        loss, focal, dice, boundary, soft_loss = criterion(studentOut, teacherOut, mask)
        train_loss += loss.item()
        train_soft_loss += soft_loss.item()
        t_focal += focal.item()
        t_dice += dice.item()
        t_boundary += boundary.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()   # check scheduler***********
    torch.cuda.empty_cache()
    gc.collect()

    avg_focal = t_focal / len(dl_t)
    avg_dice = t_dice / len(dl_t)
    avg_boundary = t_boundary / len(dl_t)
    avg_soft = train_soft_loss / len(dl_t)

    # criterion.update_epoch_losses(avg_focal, avg_dice, avg_boundary, avg_soft)  # update weights

    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch: {epoch + 1}/100, loss on training set: {train_loss / len(dl_t):.4f}, soft loss: {train_soft_loss / len(dl_t):.4f}, focal: {avg_focal:.4f}, dice: {avg_dice:.4f}, boundary: {avg_boundary:.4f}, current lr: {current_lr:.6f}')
    student.eval()
    total_dsc, total_nsd, total_hd, total_time = 0, 0, 0, 0
    with torch.no_grad():
        for enhanced, img, mask in tqdm(dl_v):
            stat_time = time.time()
            y_pred = student(img, enhanced)
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
            torch.save(student.state_dict(), f"./segModel_student.pth")
            max_dsc = avg_dsc
            print(f"********Epoch: {epoch + 1}, model's parameters saved!********")
    student.train()
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(dl_t),
            'train_soft_loss': train_soft_loss / len(dl_t),
            'valid_dsc': avg_dsc,
            'valid_nsd': avg_nsd,
            'valid_hd95': avg_hd95,
        })

