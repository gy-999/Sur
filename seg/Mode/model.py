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
        # Dual-pool channel attention
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))

        return x * spatial_att, spatial_att


# New: lightweight feature selection module
class FeatureSelectionGate(nn.Module):
    def __init__(self, channels):
        super(FeatureSelectionGate, self).__init__()
        # Very lightweight channel-wise weight learner
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 8), 1),  # further compress to reduce computation
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

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, 32, dilation_rate=2, split_ratio=0.9)
        self.encoder2 = ResidualBlock(32, 64, dilation_rate=3, split_ratio=0.8)
        self.encoder3 = ResidualBlock(64, 128, dilation_rate=4, split_ratio=0.6)

        # Bottleneck
        self.bottleneck = ResidualBlock(128, 256, dilation_rate=4, split_ratio=1)

        # Decoder
        self.decoder3 = ResidualBlock(256 + 128, 128, dilation_rate=3, split_ratio=0.9)
        self.decoder2 = ResidualBlock(128 + 64, 64, dilation_rate=2, split_ratio=0.8)
        self.decoder1 = ResidualBlock(64 + 32, 32, dilation_rate=1, split_ratio=0.7)

        # Attention modules
        self.attn3 = LightCBAM(128, reduction_ratio=4)
        self.attn2 = LightCBAM(64, reduction_ratio=2)
        self.attn1 = LightCBAM(32, reduction_ratio=2)

        # New: skip-connection feature selection gates
        self.skip_gate3 = FeatureSelectionGate(128)
        self.skip_gate2 = FeatureSelectionGate(64)
        self.skip_gate1 = FeatureSelectionGate(32)

        self.pool = nn.MaxPool2d(2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Small convolution block to refine features before final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x, return_attentions=False):
        attentions = []

        # Encoder path
        enc1 = self.encoder1(x)
        x = self.pool(enc1)

        enc2 = self.encoder2(x)
        x = self.pool(enc2)

        enc3 = self.encoder3(x)
        x = self.pool(enc3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path - add feature selection
        x = self.upsample3(x)
        if x.size(2) != enc3.size(2) or x.size(3) != enc3.size(3):
            x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=True)

        enc3_att, att3 = self.attn3(enc3)
        enc3_selected = self.skip_gate3(enc3_att)  # select important features
        attentions.append(att3)
        x = torch.cat([x, enc3_selected], dim=1)  # use the selected features
        x = self.decoder3(x)

        x = self.upsample2(x)
        if x.size(2) != enc2.size(2) or x.size(3) != enc2.size(3):
            x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)

        enc2_att, att2 = self.attn2(enc2)
        enc2_selected = self.skip_gate2(enc2_att)  # select important features
        attentions.append(att2)
        x = torch.cat([x, enc2_selected], dim=1)  # use the selected features
        x = self.decoder2(x)

        x = self.upsample1(x)
        if x.size(2) != enc1.size(2) or x.size(3) != enc1.size(3):
            x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)

        enc1_att, att1 = self.attn1(enc1)
        enc1_selected = self.skip_gate1(enc1_att)  # select important features
        attentions.append(att1)
        x = torch.cat([x, enc1_selected], dim=1)  # use the selected features
        x = self.decoder1(x)

        out = self.final_conv(x)

        if return_attentions:
            return out, attentions
        return out