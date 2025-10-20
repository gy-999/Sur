import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(UNet, self).__init__()

        # 编码器 - 3层
        self.encoder1 = self.double_conv(in_channels, 32)
        self.encoder2 = self.double_conv(32, 64)
        self.encoder3 = self.double_conv(64, 128)

        # 瓶颈层
        self.bottleneck = self.double_conv(128, 256)

        # 解码器 - 3层
        self.decoder3 = self.double_conv(256 + 128, 128)
        self.decoder2 = self.double_conv(128 + 64, 64)
        self.decoder1 = self.double_conv(64 + 32, 32)

        # 上下采样
        self.pool = nn.MaxPool2d(2)
        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.final_conv = nn.Conv2d(32, out_channels, 1)

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器路径
        enc1 = self.encoder1(x)  # [B, 32, 180, 180]
        x = self.pool(enc1)  # [B, 32, 90, 90]

        enc2 = self.encoder2(x)  # [B, 64, 90, 90]
        x = self.pool(enc2)  # [B, 64, 45, 45]

        enc3 = self.encoder3(x)  # [B, 128, 45, 45]
        x = self.pool(enc3)  # [B, 128, 22, 22] (180->90->45->22)

        # 瓶颈层
        x = self.bottleneck(x)  # [B, 256, 22, 22]

        # 解码器路径
        x = self.upsample3(x)  # [B, 256, 44, 44]

        # 使用插值确保尺寸匹配
        if x.size(2) != enc3.size(2) or x.size(3) != enc3.size(3):
            x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, enc3], dim=1)  # [B, 256+128=384, 45, 45]
        x = self.decoder3(x)  # [B, 128, 45, 45]

        x = self.upsample2(x)  # [B, 128, 90, 90]

        # 使用插值确保尺寸匹配
        if x.size(2) != enc2.size(2) or x.size(3) != enc2.size(3):
            x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, enc2], dim=1)  # [B, 128+64=192, 90, 90]
        x = self.decoder2(x)  # [B, 64, 90, 90]

        x = self.upsample1(x)  # [B, 64, 180, 180]

        # 使用插值确保尺寸匹配
        if x.size(2) != enc1.size(2) or x.size(3) != enc1.size(3):
            x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, enc1], dim=1)  # [B, 64+32=96, 180, 180]
        x = self.decoder1(x)  # [B, 32, 180, 180]

        out = self.final_conv(x)  # [B, 4, 180, 180]

        return out