# # models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCBAM(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(LightCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 双池化通道注意力
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))

        return x * spatial_att, spatial_att


# 新增：轻量级特征选择模块
class FeatureSelectionGate(nn.Module):
    def __init__(self, channels):
        super(FeatureSelectionGate, self).__init__()
        # 极轻量的通道权重学习
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 8), 1),  # 进一步压缩减少计算
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 8), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.gate(x)
        return x * weights


class DilatedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, split_ratio):
        super().__init__()
        self.split_ratio = split_ratio
        self.split_channels = int(in_channels * split_ratio)

        self.depthwise = nn.Conv2d(
            self.split_channels, self.split_channels,
            kernel_size=3, padding=dilation_rate,
            dilation=dilation_rate, groups=self.split_channels, bias=False
        )

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        attention_reduction = max(4, in_channels // 16)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, attention_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.split_ratio < 1.0:
            x_split1 = x[:, :self.split_channels, :, :]
            x_split2 = x[:, self.split_channels:, :, :]

            x_split1 = self.depthwise(x_split1)
            x_combined = torch.cat([x_split1, x_split2], dim=1)
        else:
            x_combined = self.depthwise(x)

        channel_weights = self.channel_attention(x)
        x_weighted = x_combined * channel_weights

        x = self.pointwise(x_weighted)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, split_ratio):
        super().__init__()
        self.conv1 = DilatedDepthwiseSeparableConv(in_channels, out_channels,
                                                   dilation_rate=min(dilation_rate, 4),
                                                   split_ratio=split_ratio)
        self.conv2 = DilatedDepthwiseSeparableConv(out_channels, out_channels,
                                                   dilation_rate=min(dilation_rate, 4),
                                                   split_ratio=split_ratio)

        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class EA_Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(EA_Unet, self).__init__()

        # 编码器
        self.encoder1 = ResidualBlock(in_channels, 32, dilation_rate=2, split_ratio=0.9)
        self.encoder2 = ResidualBlock(32, 64, dilation_rate=3, split_ratio=0.8)
        self.encoder3 = ResidualBlock(64, 128, dilation_rate=4, split_ratio=0.6)

        # 瓶颈层
        self.bottleneck = ResidualBlock(128, 256, dilation_rate=4, split_ratio=1)

        # 解码器
        self.decoder3 = ResidualBlock(256 + 128, 128, dilation_rate=3, split_ratio=0.9)
        self.decoder2 = ResidualBlock(128 + 64, 64, dilation_rate=2, split_ratio=0.8)
        self.decoder1 = ResidualBlock(64 + 32, 32, dilation_rate=1, split_ratio=0.7)

        # 注意力机制
        self.attn3 = LightCBAM(128, reduction_ratio=4)
        self.attn2 = LightCBAM(64, reduction_ratio=2)
        self.attn1 = LightCBAM(32, reduction_ratio=2)

        # 新增：跳跃连接特征选择门
        self.skip_gate3 = FeatureSelectionGate(128)
        self.skip_gate2 = FeatureSelectionGate(64)
        self.skip_gate1 = FeatureSelectionGate(32)

        self.pool = nn.MaxPool2d(2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 最终输出前添加一个小的卷积层细化特征
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x, return_attentions=False):
        attentions = []

        # 编码器路径
        enc1 = self.encoder1(x)
        x = self.pool(enc1)

        enc2 = self.encoder2(x)
        x = self.pool(enc2)

        enc3 = self.encoder3(x)
        x = self.pool(enc3)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器路径 - 添加特征选择
        x = self.upsample3(x)
        if x.size(2) != enc3.size(2) or x.size(3) != enc3.size(3):
            x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=True)

        enc3_att, att3 = self.attn3(enc3)
        enc3_selected = self.skip_gate3(enc3_att)  # 选择重要特征
        attentions.append(att3)
        x = torch.cat([x, enc3_selected], dim=1)  # 使用选择后的特征
        x = self.decoder3(x)

        x = self.upsample2(x)
        if x.size(2) != enc2.size(2) or x.size(3) != enc2.size(3):
            x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)

        enc2_att, att2 = self.attn2(enc2)
        enc2_selected = self.skip_gate2(enc2_att)  # 选择重要特征
        attentions.append(att2)
        x = torch.cat([x, enc2_selected], dim=1)  # 使用选择后的特征
        x = self.decoder2(x)

        x = self.upsample1(x)
        if x.size(2) != enc1.size(2) or x.size(3) != enc1.size(3):
            x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)

        enc1_att, att1 = self.attn1(enc1)
        enc1_selected = self.skip_gate1(enc1_att)  # 选择重要特征
        attentions.append(att1)
        x = torch.cat([x, enc1_selected], dim=1)  # 使用选择后的特征
        x = self.decoder1(x)

        out = self.final_conv(x)

        if return_attentions:
            return out, attentions
        return out
# class LightCBAM(nn.Module):
#     def __init__(self, channels, reduction_ratio=4):
#         super(LightCBAM, self).__init__()
#         self.channel_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction_ratio, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction_ratio, channels, 1),
#             nn.Sigmoid()
#         )
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(2, 1, 3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         channel_att = self.channel_att(x)
#         x_channel = x * channel_att
#
#         avg_out = torch.mean(x_channel, dim=1, keepdim=True)
#         max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
#         spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
#
#         return x * spatial_att, spatial_att
#
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
#                                  padding=1, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.GroupNorm(4, out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
#         self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
#         self.shortcut = nn.Conv2d(in_channels, out_channels,
#                                 kernel_size=1) if in_channels != out_channels else nn.Identity()
#
#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x + residual
#
# class SparseDepthwiseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, ratio=0.5):
#         super().__init__()
#         self.ratio = ratio
#
#         # 深度卷积部分
#         self.depthwise = nn.Conv2d(
#             in_channels, in_channels, kernel_size,
#             padding=kernel_size // 2, groups=in_channels, bias=False
#         )
#
#         # 动态稀疏注意力
#         self.attn = nn.Sequential(
#             nn.Conv2d(in_channels, max(4, in_channels // 16), kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(max(4, in_channels // 16), 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # 点卷积部分
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#
#         # 归一化与激活
#         self.bn = nn.GroupNorm(4, out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # 生成动态稀疏掩码
#         attn_map = self.attn(x)
#         B, _, H, W = attn_map.shape
#
#         # 选择Top-K重要位置
#         k = int(self.ratio * H * W)
#         _, indices = attn_map.view(B, -1).topk(k, dim=-1)
#         mask = torch.zeros_like(attn_map).view(B, -1)
#         mask.scatter_(1, indices, 1.0)
#         mask = mask.view(B, 1, H, W)
#
#         # 应用稀疏深度卷积
#         sparse_x = x * mask
#         x = self.depthwise(sparse_x)
#
#         # 全连接点卷积
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)
#
# class HybridResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=0.7):
#         super().__init__()
#         # 第一层：混合卷积
#         self.conv1 = SparseDepthwiseConv(in_channels, out_channels, ratio=ratio)
#
#         # 第二层：标准深度可分离卷积（保持稳定性）
#         self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
#
#         # 快捷连接
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.GroupNorm(4, out_channels)
#         ) if in_channels != out_channels else nn.Identity()
#
#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return F.relu(x + residual)
#
#
# class EfficientUNet(nn.Module):
#     def __init__(self, in_channels=4, out_channels=4):
#         super(EfficientUNet, self).__init__()
#
#         # 编码器：浅层使用高比率，深层使用低比率
#         self.encoder1 = HybridResidualBlock(in_channels, 32, ratio=0.8)
#         self.encoder2 = HybridResidualBlock(32, 64, ratio=0.7)
#         self.encoder3 = HybridResidualBlock(64, 128, ratio=0.6)
#
#         # 瓶颈层：标准深度可分离卷积（保持信息完整性）
#         self.bottleneck = DepthwiseSeparableConv(128, 256)
#
#         # 解码器：逐步增加稀疏比率
#         self.decoder3 = HybridResidualBlock(256 + 128, 128, ratio=0.7)
#         self.decoder2 = HybridResidualBlock(128 + 64, 64, ratio=0.8)
#         self.decoder1 = HybridResidualBlock(64 + 32, 32, ratio=0.9)
#
#         # 其余组件保持不变...
#         self.attn2 = LightCBAM(64)
#         self.attn3 = LightCBAM(128)
#         self.pool = nn.MaxPool2d(2)
#         self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
#
#     def forward(self, x, return_attentions=False):
#         attentions = []
#
#         enc1 = self.encoder1(x)
#         x = self.pool(enc1)
#         enc2 = self.encoder2(x)
#         x = self.pool(enc2)
#         enc3 = self.encoder3(x)
#         x = self.pool(enc3)
#         x = self.bottleneck(x)
#         x = self.upsample3(x)
#
#         enc3_att, att3 = self.attn3(enc3)
#         attentions.append(att3)
#         x = torch.cat([x, enc3_att], dim=1)
#         x = self.decoder3(x)
#         x = self.upsample2(x)
#
#         enc2_att, att2 = self.attn2(enc2)
#         attentions.append(att2)
#         x = torch.cat([x, enc2_att], dim=1)
#         x = self.decoder2(x)
#         x = self.upsample1(x)
#         x = torch.cat([x, enc1], dim=1)
#         x = self.decoder1(x)
#         out = self.final_conv(x)
#
#         if return_attentions:
#             return out, attentions
#         return out
