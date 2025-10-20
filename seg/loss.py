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

        # 将 target 张量的形状从 [B, H, W, 1] 改为 [B, H, W]
        target = target.squeeze(-1)

        ce_loss = F.cross_entropy(input, target, weight=self.weight)

        input_prob = F.softmax(input, dim=1)
        target_onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()

        # 排除背景类（class 0）
        input_prob = input_prob[:, 1:, :, :]  # Shape: (B, C-1, H, W)
        target_onehot = target_onehot[:, 1:, :, :]

        # 计算Dice
        input_flat = rearrange(input_prob, 'b c h w -> b c (h w)')
        target_flat = rearrange(target_onehot, 'b c h w -> b c (h w)')

        intersection = torch.sum(input_flat * target_flat, dim=2)
        cardinality = torch.sum(input_flat + target_flat, dim=2)
        dice = (2. * intersection + smooth) / (cardinality + smooth)
        dice_loss = 1 - dice.mean()

        # 对于小肿瘤区域加权，尤其是'Necrosis'区域
        weighted_dice_loss = dice_loss * self.alpha + ce_loss * (1 - self.alpha)
        return weighted_dice_loss


def multiclass_dice(output, target, n_classes, eps=1e-5):
    # 假设输出是 (B, C, H, W)
    pred = torch.argmax(output, dim=1)  # 选出最大概率的类别, 得到 (B, H, W)

    # 确保 target 是 (B, H, W)，去掉最后一个维度
    target = target.squeeze(-1)  # 如果 target 形状是 (B, H, W, 1)，那么变成 (B, H, W)

    # 将预测值和目标转换为 one-hot 编码格式
    pred_onehot = F.one_hot(pred, n_classes).permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)
    target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)

    # 计算所有类的 intersection 和 union（排除背景类 class 0）
    inter = torch.sum(pred_onehot[:, 1:] * target_onehot[:, 1:], dim=(2, 3))  # Shape: (B, C-1)
    union = torch.sum(pred_onehot[:, 1:], dim=(2, 3)) + torch.sum(target_onehot[:, 1:], dim=(2, 3))

    dice = (2. * inter + eps) / (union + eps)  # Shape: (B, C-1)
    dice_scores = dice.mean(dim=0).tolist()  # 按类别平均

    return dice_scores

