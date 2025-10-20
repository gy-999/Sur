

import torch
import torch.nn as nn
import torch.nn.functional as F
class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels=None):
        super(AttentionGate, self).__init__()
        # 如果没有指定中间通道数，使用x_channels
        if inter_channels is None:
            inter_channels = x_channels // 2

        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """
        g: 来自解码器的门控信号 [B, g_channels, H, W]
        x: 来自编码器的跳跃连接 [B, x_channels, H, W]
        """
        # 下采样门控信号以匹配x的尺寸
        if g.size(2) != x.size(2) or g.size(3) != x.size(3):
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)  # [B, inter_channels, H, W]
        x1 = self.W_x(x)  # [B, inter_channels, H, W]

        psi = self.relu(g1 + x1)  # [B, inter_channels, H, W]
        psi = self.psi(psi)  # [B, 1, H, W]
        att = self.sigmoid(psi)  # [B, 1, H, W]

        return x * att  # [B, x_channels, H, W]


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(AttentionUNet, self).__init__()

        # Encoder - 3层
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.bottleneck = self._block(128, 256)  # 瓶颈层

        # Attention Gates - 3个注意力门
        # attn3: g来自上采样后的bottleneck(128), x来自enc3(128)
        self.attn3 = AttentionGate(g_channels=128, x_channels=128, inter_channels=64)
        # attn2: g来自上采样后的dec3(64), x来自enc2(64)
        self.attn2 = AttentionGate(g_channels=64, x_channels=64, inter_channels=32)
        # attn1: g来自上采样后的dec2(32), x来自enc1(32)
        self.attn1 = AttentionGate(g_channels=32, x_channels=32, inter_channels=16)

        # Decoder - 修正通道数
        self.dec3 = self._block(128 + 128, 128)  # 128(上采样) + 128(注意力) = 256 -> 128
        self.dec2 = self._block(64 + 64, 64)     # 64 + 64 = 128 -> 64
        self.dec1 = self._block(32 + 32, 32)     # 32 + 32 = 64 -> 32

        # Pooling & Upsampling
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder
        e1 = self.enc1(x)  # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 128, H/4, W/4]
        b = self.bottleneck(self.pool(e3))  # [B, 256, H/8, W/8] (180->90->45->22)

        # Decoder with attention
        # Level 3
        d3_up = self.up3(b)  # [B, 128, H/4, W/4]

        # 确保尺寸匹配
        if d3_up.size(2) != e3.size(2) or d3_up.size(3) != e3.size(3):
            d3_up = F.interpolate(d3_up, size=e3.shape[2:], mode='bilinear', align_corners=False)

        att3 = self.attn3(g=d3_up, x=e3)  # [B, 128, H/4, W/4]
        d3 = torch.cat([d3_up, att3], dim=1)  # [B, 128+128=256, H/4, W/4]
        d3 = self.dec3(d3)  # [B, 128, H/4, W/4]

        # Level 2
        d2_up = self.up2(d3)  # [B, 64, H/2, W/2]

        # 确保尺寸匹配
        if d2_up.size(2) != e2.size(2) or d2_up.size(3) != e2.size(3):
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='bilinear', align_corners=False)

        att2 = self.attn2(g=d2_up, x=e2)  # [B, 64, H/2, W/2]
        d2 = torch.cat([d2_up, att2], dim=1)  # [B, 64+64=128, H/2, W/2]
        d2 = self.dec2(d2)  # [B, 64, H/2, W/2]

        # Level 1
        d1_up = self.up1(d2)  # [B, 32, H, W]

        # 确保尺寸匹配
        if d1_up.size(2) != e1.size(2) or d1_up.size(3) != e1.size(3):
            d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode='bilinear', align_corners=False)

        att1 = self.attn1(g=d1_up, x=e1)  # [B, 32, H, W]
        d1 = torch.cat([d1_up, att1], dim=1)  # [B, 32+32=64, H, W]
        d1 = self.dec1(d1)  # [B, 32, H, W]

        # 最终输出
        return self.final_conv(d1)