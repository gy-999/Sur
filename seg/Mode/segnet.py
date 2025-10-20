
# 2. SegNet (Badrinarayanan et al., 2017)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(SegNet, self).__init__()

        # 编码器 (VGG16风格)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 解码器（新增独立的解码器层）
        self.dec4 = self.conv_block(512, 256)  # 不再共享权重
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, 3, padding=1)

        # 池化/上采样
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        size1 = e1.size()
        e1, idx1 = self.pool(e1)

        e2 = self.enc2(e1)
        size2 = e2.size()
        e2, idx2 = self.pool(e2)

        e3 = self.enc3(e2)
        size3 = e3.size()
        e3, idx3 = self.pool(e3)

        e4 = self.enc4(e3)
        size4 = e4.size()
        e4, idx4 = self.pool(e4)

        # 解码路径（使用独立的解码器层）
        d4 = self.unpool(e4, idx4, output_size=size4)
        d4 = self.dec4(d4)  # 使用解码器层而不是编码器层

        d3 = self.unpool(d4, idx3, output_size=size3)
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2, output_size=size2)
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1, output_size=size1)
        d1 = self.dec1(d1)

        return d1