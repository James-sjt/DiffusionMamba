from segDataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from TransUNet import VisionTransformer
import os
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
import random
from TransUNet import CONFIGS as CONFIGS_ViT_seg
import sys
sys.path.append('../../')
from lossfunction import Loss
from metric import hd95, nsd, dsc
import time
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    # net.load_from(weights=np.load(config_vit.pretrained_path))

    ds_t = ImageDataset('train', device, imgType="original")
    ds_v = ImageDataset('valid', device, imgType="original")

    dl_t = DataLoader(ds_t, batch_size=8, shuffle=True)
    dl_v = DataLoader(ds_v, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    batches_per_epoch = len(dl_t)
    total_steps = 100 * batches_per_epoch
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5
    )
    max_dsc = 0
    criterion = Loss()
    model.train()
    for epoch in range(100):
        total_loss, t_focal, t_dice, t_boundary = 0., 0., 0., 0.
        for img, mask in tqdm(dl_t):
            pred = model(img)
            loss, focal, dice, boundary = criterion(pred, mask)

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
                torch.save(model.state_dict(), f"./transUNet.pth")
                max_dsc = avg_dsc
                print(f"********Epoch: {epoch + 1}, model's parameters saved!********")
        model.train()
