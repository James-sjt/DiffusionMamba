import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)  # convert logits → probs
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class SobelBoundaryLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=device)
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)  # shape: (1,1,3,3)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
    
    def forward(self, logits, targets):
        # Convert logits to probabilities
        pred = torch.sigmoid(logits)
        gt = targets.float()
        
        # Convolve with Sobel kernels
        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        gt_dx = F.conv2d(gt, self.sobel_x, padding=1)
        gt_dy = F.conv2d(gt, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        pred_edge = torch.abs(pred_dx) + torch.abs(pred_dy)
        gt_edge = torch.abs(gt_dx) + torch.abs(gt_dy)
        
        # L1 loss between predicted and GT edges
        loss = F.l1_loss(pred_edge, gt_edge)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

class Loss(nn.Module):
    def __init__(self, temp=2.0, eps=1e-8):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice = DiceLoss()
        self.boundary_loss = SobelBoundaryLoss()

        self.temp = temp
        self.eps = eps

        self.prev_losses = None
        self.curr_losses = None

    @torch.no_grad()
    def update_epoch_losses(self, focal, dice, boundary):
        self.prev_losses = self.curr_losses
        self.curr_losses = torch.tensor([focal, dice, boundary], device='cpu') 

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets) / 0.02
        dice_loss = self.dice(inputs, targets) / 0.3
        boundary_loss = self.boundary_loss(inputs, targets) / 0.02

        if self.prev_losses is None or self.curr_losses is None:
            weights = torch.ones(3, device=inputs.device) / 3.0
        else:
            r = self.curr_losses / (self.prev_losses + self.eps)
            weights = F.softmax(r / self.temp, dim=0).to(inputs.device)
        
        return (
                weights[0] * focal_loss
                + weights[1] * dice_loss
                + weights[2] * boundary_loss
        ), focal_loss * 0.02, dice_loss * 0.3, boundary_loss * 0.02
