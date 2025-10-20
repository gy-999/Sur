import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class DenseAtrousConv(nn.Module):
    """密集空洞卷积模块"""

    def __init__(self, in_channels, dilation_rates=[1, 2, 3, 4]):
        super().__init__()
        self.branches = nn.ModuleList()

        for rate in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # 密集连接所有分支输出
        combined = torch.cat(branch_outputs, dim=1)
        return self.fusion(combined) + x


class ResidualMultiKernelPooling(nn.Module):
    """残差多核池化模块"""

    def __init__(self, in_channels):
        super().__init__()
        self.branches = nn.ModuleList()
        pool_sizes = [1, 2, 3, 6]

        for size in pool_sizes:
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        branch_outputs = []

        for branch in self.branches:
            out = branch(x)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            branch_outputs.append(out)

        combined = torch.cat(branch_outputs, dim=1)
        return self.fusion(combined) + x


class CE_Net(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(CE_Net, self).__init__()

        # Encoder - 3层
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # CE-Net特有的瓶颈层
        self.dac_block = DenseAtrousConv(128)  # 输入128通道
        self.rmp_block = ResidualMultiKernelPooling(128)  # 输入128通道

        # 瓶颈层扩展通道
        self.bottleneck_expand = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),  # 输出256通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder - 3层
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)

        # 上采样层 - 3个
        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.final_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder
        enc1 = self.encoder1(x)  # [B, 32, H, W]
        x = self.pool(enc1)  # [B, 32, H/2, W/2]

        enc2 = self.encoder2(x)  # [B, 64, H/2, W/2]
        x = self.pool(enc2)  # [B, 64, H/4, W/4]

        enc3 = self.encoder3(x)  # [B, 128, H/4, W/4]
        x = self.pool(enc3)  # [B, 128, H/8, W/8] (180->90->45->22)

        # CE-Net特有的瓶颈处理
        x = self.dac_block(x)  # 密集空洞卷积 [B, 128, H/8, W/8]
        x = self.rmp_block(x)  # 残差多核池化 [B, 128, H/8, W/8]
        x = self.bottleneck_expand(x)  # [B, 256, H/8, W/8]

        # Decoder
        x = self.upsample3(x)  # [B, 256, H/4, W/4]

        # 确保尺寸匹配
        if x.size(2) != enc3.size(2) or x.size(3) != enc3.size(3):
            x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, enc3], dim=1)  # [B, 256+128, H/4, W/4]
        x = self.decoder3(x)  # [B, 128, H/4, W/4]

        x = self.upsample2(x)  # [B, 128, H/2, W/2]

        # 确保尺寸匹配
        if x.size(2) != enc2.size(2) or x.size(3) != enc2.size(3):
            x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, enc2], dim=1)  # [B, 128+64, H/2, W/2]
        x = self.decoder2(x)  # [B, 64, H/2, W/2]

        x = self.upsample1(x)  # [B, 64, H, W]

        # 确保尺寸匹配
        if x.size(2) != enc1.size(2) or x.size(3) != enc1.size(3):
            x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, enc1], dim=1)  # [B, 64+32, H, W]
        x = self.decoder1(x)  # [B, 32, H, W]

        # 最终输出
        out = self.final_conv(x)
        return out