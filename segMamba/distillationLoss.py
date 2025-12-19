import torch
import torch.nn.functional as F
from lossfunction import DiceLoss, FocalLoss, SobelBoundaryLoss
import torch.nn as nn

def softLoss(pred_teacher, pred_student, mask):  # pred_teacher, pred_student: sigmoid
    seg_teacher = (pred_teacher > 0.5).float()
    seg_student = (pred_student > 0.5).float()

    teacher_correct = (seg_teacher == mask).float()
    student_wrong = (seg_student != mask).float()

    focus_region = (teacher_correct * student_wrong)

    diff = (pred_teacher - pred_student) ** 2
    numerator = torch.sum(diff * focus_region)
    denominator = torch.sum(focus_region) + 1e-8

    soft_loss = numerator / denominator
    return soft_loss


class DistillationLoss(nn.Module):
    def __init__(self, T=2.0, eps=1e-8):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss(smooth=eps)
        self.boundary_loss = SobelBoundaryLoss()

        self.T = T
        self.eps = eps

        # self.prev_losses = None
        # self.curr_losses = None

    # @torch.no_grad()
    # def update_epoch_losses(self, focal, dice, boundary, soft):
    #     self.prev_losses = self.curr_losses
    #     self.curr_losses = torch.tensor([focal, dice, boundary, soft], device='cpu')

    def forward(self, y_student, y_teacher, truths):
        focal_loss = self.focal_loss(y_student, truths) / 0.004
        dice_loss = self.dice_loss(y_student, truths) / 0.17
        boundary_loss = self.boundary_loss(y_student, truths) / 0.013

        scaled_p_student = torch.sigmoid(y_student / self.T)
        scaled_p_teacher = torch.sigmoid(y_teacher / self.T)

        soft_loss = softLoss(scaled_p_teacher, scaled_p_student, truths) / 0.1

        # if self.prev_losses is None or self.curr_losses is None:
        weights = torch.ones(4, device=y_student.device) / 4
        # else:
        #     r = self.curr_losses / (self.prev_losses + self.eps)
        #     weights = F.softmax(r / self.T, dim=0).to(y_student.device)

        return (weights[0] * focal_loss
                + weights[1] * dice_loss
                + weights[2] * boundary_loss
                + weights[3] * soft_loss), focal_loss * 0.004, dice_loss * 0.17, boundary_loss * 0.013, soft_loss * 0.1

