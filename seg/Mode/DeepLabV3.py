import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
import torch.nn.functional as F

# 定义 DeepLabV3+ 模型
class DeepLabV3(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(DeepLabV3, self).__init__()

        # 使用 torchvision 提供的 DeepLabV3 模型
        self.deeplabv3 = segmentation.deeplabv3_resnet101(pretrained=False)  # 不使用预训练，确保公平对比

        # 修改输入层，以适应自定义输入通道数
        original_conv1 = self.deeplabv3.backbone.conv1
        self.deeplabv3.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # 初始化新的第一层卷积
        if in_channels == 3:
            # 如果是3通道，可以使用预训练权重的一部分
            with torch.no_grad():
                self.deeplabv3.backbone.conv1.weight[:, :3] = original_conv1.weight
        else:
            # 其他通道数，使用Kaiming初始化
            nn.init.kaiming_normal_(self.deeplabv3.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # 修改ASPP模块的输出通道数
        self.deeplabv3.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)

        # 添加最终上采样层以确保输出尺寸与输入一致
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 保存输入尺寸
        input_size = x.shape[2:]

        # 前向传播
        out = self.deeplabv3(x)['out']

        # 上采样到原始输入尺寸
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out


# 轻量级版本，使用ResNet50 backbone
class DeepLabV3_Lite(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(DeepLabV3_Lite, self).__init__()

        # 使用ResNet50 backbone以减少计算量
        self.deeplabv3 = segmentation.deeplabv3_resnet50(pretrained=False)

        # 修改输入层
        original_conv1 = self.deeplabv3.backbone.conv1
        self.deeplabv3.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # 初始化
        nn.init.kaiming_normal_(self.deeplabv3.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # 修改输出层
        self.deeplabv3.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        out = self.deeplabv3(x)['out']

        # 上采样到原始输入尺寸
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out