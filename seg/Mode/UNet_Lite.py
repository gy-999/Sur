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
# 假设之前定义了 ConvBNReLU, SepConv, Conv1x1, InvertedResidualLite 如前

class UNetPlusPlus_Lite(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(in_channels, base_ch), SepConv(base_ch, base_ch))
        self.enc2 = nn.Sequential(ConvBNReLU(base_ch, base_ch*2), SepConv(base_ch*2, base_ch*2))
        self.enc3 = nn.Sequential(ConvBNReLU(base_ch*2, base_ch*4), SepConv(base_ch*4, base_ch*4))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBNReLU(base_ch*4, base_ch*8), SepConv(base_ch*8, base_ch*8))

        # projections
        self.proj_enc1 = Conv1x1(base_ch, base_ch)
        self.proj_enc2 = Conv1x1(base_ch*2, base_ch*2)
        self.proj_enc3 = Conv1x1(base_ch*4, base_ch*4)
        self.proj_bottleneck = Conv1x1(base_ch*8, base_ch*8)

        # 当我们 cat 后通道数为 sum(...)，先用 1x1 降维到合适通道再用 SepConv
        # x2_1: cat(u3_0(p3 up -> 4C), p2(4C)) -> 8C  -> project -> 4C -> SepConv(4C->4C)
        self.project_x2_1 = Conv1x1(base_ch*8, base_ch*4)
        self.conv_x2_1 = SepConv(base_ch*4, base_ch*4)

        # x1_1: cat(u2_0(p2 up -> 2C), p1(2C)) -> 4C -> project -> 2C -> SepConv(2C->2C)
        self.project_x1_1 = Conv1x1(base_ch*4, base_ch*2)
        self.conv_x1_1 = SepConv(base_ch*2, base_ch*2)

        # x1_2: cat(up(x2_1)->2C, x1_1->2C) -> 4C -> project -> 2C -> SepConv(2C->2C)
        self.project_x1_2 = Conv1x1(base_ch*4, base_ch*2)
        self.conv_x1_2 = SepConv(base_ch*2, base_ch*2)

        # decoder convs (同理：cat 后先投影)
        # dec3: cat(up(bottleneck)->4C, enc3->4C) -> 8C -> project -> 4C -> SepConv(4C->4C)
        self.project_dec3 = Conv1x1(base_ch*8, base_ch*4)
        self.dec3 = SepConv(base_ch*4, base_ch*4)

        # dec2: cat(up(dec3)->2C, enc2->2C) -> 4C -> project -> 2C -> SepConv(2C->2C)
        self.project_dec2 = Conv1x1(base_ch*4, base_ch*2)
        self.dec2 = SepConv(base_ch*2, base_ch*2)

        # dec1: cat(up(dec2)->C, enc1->C) -> 2C -> project -> C -> SepConv(C->C)
        self.project_dec1 = Conv1x1(base_ch*2, base_ch)
        self.dec1 = SepConv(base_ch, base_ch)

        self.final_conv = nn.Conv2d(base_ch, out_channels, 1)

    def _upsample_to(self, src, target):
        if src.shape[2:] != target.shape[2:]:
            return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)
        return src

    def forward(self, x):
        x0_0 = self.enc1(x)
        x1_0 = self.enc2(self.pool(x0_0))
        x2_0 = self.enc3(self.pool(x1_0))
        x3_0 = self.bottleneck(self.pool(x2_0))

        p0 = self.proj_enc1(x0_0)
        p1 = self.proj_enc2(x1_0)
        p2 = self.proj_enc3(x2_0)
        p3 = self.proj_bottleneck(x3_0)

        # x2_1
        u3_0 = self._upsample_to(p3, p2)
        cat22 = torch.cat([u3_0, p2], dim=1)
        cat22 = self.proj_cat22(cat22)  # Conv1x1 将通道降到 256
        cat22 = self.conv_x2_1(cat22)            # -> 4C
        x2_1 = self.conv_x2_1(cat22)                   # -> 4C

        # x1_1
        u2_0 = self._upsample_to(p2, p1)
        cat11 = torch.cat([u2_0, p1], dim=1)           # -> 4C
        cat11 = self.project_x1_1(cat11)               # -> 2C
        x1_1 = self.conv_x1_1(cat11)                   # -> 2C

        # x1_2 (aggregate deeper)
        u2_1 = self._upsample_to(x2_1, x1_1)
        cat12 = torch.cat([u2_1, x1_1], dim=1)         # -> 4C
        cat12 = self.project_x1_2(cat12)               # -> 2C
        x1_2 = self.conv_x1_2(cat12)                   # -> 2C

        # decoder: use x2_1 as decoded top feature
        u_dec3 = self._upsample_to(p3, p2)             # up bottleneck to x2 size (we could also use x2_1)
        cat_dec3 = torch.cat([u_dec3, p2], dim=1)      # -> 8C
        cat_dec3 = self.project_dec3(cat_dec3)         # -> 4C
        d3 = self.dec3(cat_dec3)                       # -> 4C

        u_dec2 = self._upsample_to(d3, p1)
        cat_dec2 = torch.cat([u_dec2, p1], dim=1)      # -> 4C
        cat_dec2 = self.project_dec2(cat_dec2)         # -> 2C
        d2 = self.dec2(cat_dec2)                       # -> 2C

        u_dec1 = self._upsample_to(d2, p0)
        cat_dec1 = torch.cat([u_dec1, p0], dim=1)      # -> 2C
        cat_dec1 = self.project_dec1(cat_dec1)         # -> C
        d1 = self.dec1(cat_dec1)                       # -> C

        out = self.final_conv(d1)
        return out