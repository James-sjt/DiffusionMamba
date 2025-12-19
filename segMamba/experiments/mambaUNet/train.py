import torch
from segDataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('../../')

from lossfunction import Loss
from mambaUNet import mambaUNet
from metric import hd95, nsd, dsc
import time
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ImageDataset('train', device, imgType="original")
datasetValid = ImageDataset('valid', device, imgType="original")
dl_t = DataLoader(dataset, batch_size=8, shuffle=True)
dl_v = DataLoader(datasetValid, batch_size=1, shuffle=False)
criterion = Loss()
model = mambaUNet().to(device)  # check cond

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
batches_per_epoch = len(dl_t)
total_steps = 100 * batches_per_epoch

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=1e-5
)
max_dsc = 0

model.train()
for epoch in range(100):
    total_loss, t_focal, t_dice, t_boundary = 0., 0., 0., 0.
    # Training Loop
    for img, mask in tqdm(dl_t):
        out = model(img)
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

    model.eval()
    total_dsc, total_nsd, total_hd, total_time = 0, 0, 0, 0
    with torch.no_grad():
        for img, mask in tqdm(dl_v):
            stat_time = time.time()
            y_pred = model(img)
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            total_time += time.time() - stat_time
            total_dsc += dsc(y_pred, mask)
            total_nsd += nsd(y_pred, mask)
            total_hd += hd95(y_pred, mask)
            del img, mask, y_pred
        torch.cuda.empty_cache()
        gc.collect()
        avg_dsc = total_dsc / len(dl_v)
        avg_nsd = total_nsd / len(dl_v)
        avg_hd95 = total_hd / len(dl_v)
        avg_time = total_time / len(dl_v)

        print(f'DSC: {avg_dsc:.4f}, NSD: {avg_nsd:.4f}, HD95: {avg_hd95:.4f}, Inference Time: {avg_time:.4f}')
        if avg_dsc > max_dsc:
            torch.save(model.state_dict(), f"./mambaUNet.pth")
            max_dsc = avg_dsc
            print(f"********Epoch: {epoch + 1}, model's parameters saved!********")
    model.train()

