import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 基础轻量化模块 ----------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SepConv(nn.Module):
    """Depthwise separable conv: depthwise -> pointwise"""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------- EfficientUNet_Lite ----------
class InvertedResidualLite(nn.Module):
    """Simple inverted bottleneck: expand -> depthwise -> project"""
    def __init__(self, in_ch, mid_ch, out_ch, kernel=3):
        super().__init__()
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.dw = nn.Conv2d(mid_ch, mid_ch, kernel, padding=kernel//2, groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.act2 = nn.ReLU(inplace=True)
        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.act1(self.bn1(self.expand(x)))
        x = self.act2(self.bn2(self.dw(x)))
        x = self.bn3(self.project(x))
        return x

class EfficientUNet_Lite(nn.Module):
    """
    Efficient-style U-Net lite.
    Use inverted residual blocks in encoder/decoder and interpolate+conv upsampling.
    """
    def __init__(self, in_channels=4, out_channels=4, base_ch=32):
        super().__init__()
        # encoder
        self.enc1 = nn.Sequential(ConvBNReLU(in_channels, base_ch), InvertedResidualLite(base_ch, base_ch*2, base_ch))
        self.enc2 = nn.Sequential(ConvBNReLU(base_ch, base_ch*2), InvertedResidualLite(base_ch*2, base_ch*4, base_ch*2))
        self.enc3 = nn.Sequential(ConvBNReLU(base_ch*2, base_ch*4), InvertedResidualLite(base_ch*4, base_ch*8, base_ch*4))
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = InvertedResidualLite(base_ch*4, base_ch*8, base_ch*8)

        # decoder: upsample (interpolate) + conv to reduce channels and then combine
        self.up_conv3 = SepConv(base_ch*8, base_ch*4)
        self.up_conv2 = SepConv(base_ch*4, base_ch*2)
        self.up_conv1 = SepConv(base_ch*2, base_ch)

        self.dec3 = SepConv(base_ch*4 + base_ch*4, base_ch*4)
        self.dec2 = SepConv(base_ch*2 + base_ch*2, base_ch*2)
        self.dec1 = SepConv(base_ch + base_ch, base_ch)

        self.final_conv = nn.Conv2d(base_ch, out_channels, 1)

    def _upsample_to(self, src, target):
        if src.shape[2:] != target.shape[2:]:
            return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)
        return src

    def forward(self, x):
        x0 = self.enc1(x)                # C
        x1 = self.enc2(self.pool(x0))    # 2C
        x2 = self.enc3(self.pool(x1))    # 4C
        x3 = self.bottleneck(self.pool(x2))  # 8C

        # upsample path
        u3 = self._upsample_to(self.up_conv3(x3), x2)  # -> 4C
        d3 = self.dec3(torch.cat([u3, x2], dim=1))     # -> 4C

        u2 = self._upsample_to(self.up_conv2(d3), x1)  # -> 2C
        d2 = self.dec2(torch.cat([u2, x1], dim=1))     # -> 2C

        u1 = self._upsample_to(self.up_conv1(d2), x0)  # -> C
        d1 = self.dec1(torch.cat([u1, x0], dim=1))     # -> C

        out = self.final_conv(d1)
        return out
