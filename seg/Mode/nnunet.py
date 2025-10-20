import torch
import torch.nn as nn
import torch.nn.functional as F


# 5. nnUNet (Isensee et al., 2021) 的2D简化版
class nnUNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(nnUNet2D, self).__init__()

        # 编码器路径 - 增加到5层（包括瓶颈层）
        self.conv1 = self.res_block(in_channels, 32)
        self.conv2 = self.res_block(32, 64)
        self.conv3 = self.res_block(64, 128)
        self.conv4 = self.res_block(128, 256)
        self.conv5 = self.res_block(256, 512)  # 新增瓶颈层

        # 解码器路径 - 增加到4层
        self.upconv4 = self.upconv(512, 256)  # 新增上采样层
        self.conv6 = self.res_block(512, 256)  # 256 + 256

        self.upconv3 = self.upconv(256, 128)
        self.conv7 = self.res_block(256, 128)  # 128 + 128

        self.upconv2 = self.upconv(128, 64)
        self.conv8 = self.res_block(128, 64)  # 64 + 64

        self.upconv1 = self.upconv(64, 32)
        self.conv9 = self.res_block(64, 32)  # 32 + 32

        self.final_conv = nn.Conv2d(32, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def res_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def upconv(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 编码
        c1 = self.conv1(x)  # [B, 32, H, W]
        p1 = self.pool(c1)  # [B, 32, H/2, W/2]

        c2 = self.conv2(p1)  # [B, 64, H/2, W/2]
        p2 = self.pool(c2)  # [B, 64, H/4, W/4]

        c3 = self.conv3(p2)  # [B, 128, H/4, W/4]
        p3 = self.pool(c3)  # [B, 128, H/8, W/8]

        c4 = self.conv4(p3)  # [B, 256, H/8, W/8]
        p4 = self.pool(c4)  # [B, 256, H/16, W/16]

        # 瓶颈层
        c5 = self.conv5(p4)  # [B, 512, H/16, W/16]

        # 解码
        u4 = self.upconv4(c5)  # [B, 256, H/8, W/8]

        # 确保尺寸匹配
        if u4.size(2) != c4.size(2) or u4.size(3) != c4.size(3):
            u4 = F.interpolate(u4, size=c4.shape[2:], mode='bilinear', align_corners=True)

        u4 = torch.cat([u4, c4], dim=1)  # [B, 512, H/8, W/8]
        c6 = self.conv6(u4)  # [B, 256, H/8, W/8]

        u3 = self.upconv3(c6)  # [B, 128, H/4, W/4]

        # 确保尺寸匹配
        if u3.size(2) != c3.size(2) or u3.size(3) != c3.size(3):
            u3 = F.interpolate(u3, size=c3.shape[2:], mode='bilinear', align_corners=True)

        u3 = torch.cat([u3, c3], dim=1)  # [B, 256, H/4, W/4]
        c7 = self.conv7(u3)  # [B, 128, H/4, W/4]

        u2 = self.upconv2(c7)  # [B, 64, H/2, W/2]

        # 确保尺寸匹配
        if u2.size(2) != c2.size(2) or u2.size(3) != c2.size(3):
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=True)

        u2 = torch.cat([u2, c2], dim=1)  # [B, 128, H/2, W/2]
        c8 = self.conv8(u2)  # [B, 64, H/2, W/2]

        u1 = self.upconv1(c8)  # [B, 32, H, W]

        # 确保尺寸匹配
        if u1.size(2) != c1.size(2) or u1.size(3) != c1.size(3):
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=True)

        u1 = torch.cat([u1, c1], dim=1)  # [B, 64, H, W]
        c9 = self.conv9(u1)  # [B, 32, H, W]

        # 最终输出，确保与输入尺寸一致
        output = self.final_conv(c9)
        if output.size(2) != H or output.size(3) != W:
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        return output