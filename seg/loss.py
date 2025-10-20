# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CombinedLoss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.6):
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.weight = weight

    def forward(self, input, target):
        smooth = 1e-5


        target = target.squeeze(-1)

        ce_loss = F.cross_entropy(input, target, weight=self.weight)

        input_prob = F.softmax(input, dim=1)
        target_onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()


        input_prob = input_prob[:, 1:, :, :]
        target_onehot = target_onehot[:, 1:, :, :]


        input_flat = rearrange(input_prob, 'b c h w -> b c (h w)')
        target_flat = rearrange(target_onehot, 'b c h w -> b c (h w)')

        intersection = torch.sum(input_flat * target_flat, dim=2)
        cardinality = torch.sum(input_flat + target_flat, dim=2)
        dice = (2. * intersection + smooth) / (cardinality + smooth)
        dice_loss = 1 - dice.mean()


        weighted_dice_loss = dice_loss * self.alpha + ce_loss * (1 - self.alpha)
        return weighted_dice_loss


def multiclass_dice(output, target, n_classes, eps=1e-5):

    pred = torch.argmax(output, dim=1)
    target = target.squeeze(-1)


    pred_onehot = F.one_hot(pred, n_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()

    inter = torch.sum(pred_onehot[:, 1:] * target_onehot[:, 1:], dim=(2, 3))
    union = torch.sum(pred_onehot[:, 1:], dim=(2, 3)) + torch.sum(target_onehot[:, 1:], dim=(2, 3))

    dice = (2. * inter + eps) / (union + eps)
    dice_scores = dice.mean(dim=0).tolist()

    return dice_scores

