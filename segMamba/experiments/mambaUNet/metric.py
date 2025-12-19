import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Experiments of Several Segmentation Models")
parser.add_argument('--model_name', type=str, default='student')


def dsc(y_pred, y_true):
    batch_size = y_pred.shape[0]
    dice = 0
    for i in range(batch_size):
        pred = y_pred[i]
        gt = y_true[i]
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt)

        if union == 0:
            dice += 1
        else:
            dice += (2. * intersection) / union

    return dice / batch_size


def compute_surface(mask):
    # ensure float
    if mask.dim() == 3 and mask.size(0) == 1:
        m = mask.squeeze(0)
    elif mask.dim() == 2:
        m = mask
    else:
        raise ValueError("mask must be [1,H,W] or [H,W]")
    m = (m > 0.5).float()  # ensure binary

    # erosion: min pooling = -max_pool2d(-x)
    m_unsq = m.unsqueeze(0)  # shape [1, H, W] for pooling
    eroded = -F.max_pool2d(-m_unsq, kernel_size=3, stride=1, padding=1).squeeze(0)
    # surface: pixels present in mask but removed by erosion (foreground boundary)
    surface = (m - eroded).abs()
    # make binary surface map
    return (surface > 0).float()  # shape [H, W]


def nsd(y_pred, y_true, tolerance=1.0, pred_threshold=0.5, eps=1e-8):
    # normalize shapes to [B, 1, H, W]
    if y_pred.dim() == 3:  # [B, H, W]
        y_pred = y_pred.unsqueeze(1)
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.size(1) != 1 or y_true.size(1) != 1:
        raise ValueError("Expected single-channel masks [B,1,H,W] or [B,H,W]")

    batch_size = y_pred.size(0)
    device = y_pred.device
    dtype = y_pred.dtype
    scores = []

    for b in range(batch_size):
        pred = y_pred[b, 0]
        gt = y_true[b, 0]

        # binarize predictions and ground truth (gt may already be 0/1)
        pred_bin = (pred > pred_threshold).float()
        gt_bin = (gt > 0.5).float()

        pred_surf = compute_surface(pred_bin)  # [H, W]
        gt_surf = compute_surface(gt_bin)

        # get coordinates (y, x) for surface pixels
        pred_pts = torch.nonzero(pred_surf, as_tuple=False)  # shape [Np, 2] (y,x)
        gt_pts = torch.nonzero(gt_surf, as_tuple=False)      # shape [Ng, 2]

        Np = pred_pts.size(0)
        Ng = gt_pts.size(0)

        # Handle empty cases
        if Np == 0 and Ng == 0:
            scores.append(torch.tensor(1.0, device=device, dtype=dtype))  # perfect match
            continue
        if Np == 0 or Ng == 0:
            scores.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue

        # convert to float and compute pairwise distances in pixel units
        pred_pts_f = pred_pts.float().to(device)
        gt_pts_f = gt_pts.float().to(device)

        # distances: for each pred point, closest gt point
        dist_pred_to_gt = torch.cdist(pred_pts_f, gt_pts_f).min(dim=1)[0]  # [Np]
        dist_gt_to_pred = torch.cdist(gt_pts_f, pred_pts_f).min(dim=1)[0]  # [Ng]

        matched_pred = (dist_pred_to_gt <= tolerance).float().sum()   # how many pred surface pts matched
        matched_gt = (dist_gt_to_pred <= tolerance).float().sum()     # how many gt surface pts matched

        TP = (matched_pred + matched_gt) / 2.0
        FP = Np - matched_pred
        FN = Ng - matched_gt

        nsd_score = TP / (TP + FP + FN + eps)
        scores.append(nsd_score)

    return torch.stack(scores).mean()


def hd95(y_pred, y_true, threshold=0.5, spacing=(1.0, 1.0), already_sigmoid=True):
    batch_size = y_pred.shape[0]
    hd95_list = []

    for b in range(batch_size):
        pred = y_pred[b, 0].cpu().numpy()
        gt = y_true[b, 0].cpu().numpy()

        # Apply sigmoid if needed
        if not already_sigmoid:
            pred = torch.sigmoid(pred)

        pred = (pred > threshold).astype(bool)
        gt = (gt > 0.5).astype(bool)

        # Case 1: both empty → perfect
        if not pred.any() and not gt.any():
            hd95_list.append(0.0)
            continue

        # Case 2: prediction is empty, but ground truth is not → bad prediction
        if not pred.any() and gt.any():
            hd95_list.append(5)
            continue

        # Case 3: ground truth is empty, but prediction is not → bad prediction
        if pred.any() and not gt.any():
            hd95_list.append(5)
            continue

        # Compute binary surfaces (edges)
        pred_surf = np.logical_xor(pred, binary_erosion(pred))
        gt_surf = np.logical_xor(gt, binary_erosion(gt))

        # Distance transforms (compute distance to the nearest background pixel)
        dist_pred = distance_transform_edt(~pred, sampling=spacing)
        dist_gt = distance_transform_edt(~gt, sampling=spacing)

        # Distances at surfaces
        surf_to_gt = dist_gt[pred_surf]
        gt_to_surf = dist_pred[gt_surf]

        # If either has no surface points, treat as mismatch
        if surf_to_gt.size == 0 or gt_to_surf.size == 0:
            hd95_list.append(np.inf)
            continue

        # Hausdorff 95: 95th percentile of symmetric distances
        try:
            hd95_val = np.percentile(np.hstack([surf_to_gt, gt_to_surf]), 95)
        except ValueError:
            hd95_list.append(np.inf)
            continue
        hd95_list.append(hd95_val)
    # Safely average across batch, ignoring ∞ cases (will happen on missed/extra objects)
    valid_vals = [v for v in hd95_list if np.isfinite(v)]
    if len(valid_vals) == 0:
        return float('inf')
    return float(np.mean(valid_vals))

def iou(y_pred, y_true):
    intersection = torch.sum(y_pred * y_true)
    if torch.sum(y_pred) + torch.sum(y_true) == 0:
        return 1
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection

    iou_score = intersection / (union + 1e-8)
    return iou_score
