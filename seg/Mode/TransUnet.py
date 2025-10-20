import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=128, embed_dim=256, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 128, H, W]
        x = self.proj(x)  # [B, 256, H/2, W/2]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H * W
        x = self.norm(x)
        return x, (H, W)  # 返回特征图尺寸


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)

        return x


class TransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(TransUNet, self).__init__()

        # CNN Encoder - 3层
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

        # 下采样层 - 3个
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        # Patch Embedding for Transformer - 调整输入通道数
        self.patch_embed = PatchEmbedding(128, 256, patch_size=2)  # 输入128通道，输出256

        # Transformer Encoder
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(256, num_heads=8) for _ in range(4)]  # 调整维度为256
        )

        # 上采样层 - 3个
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # CNN Decoder - 3层
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

        self.final_conv = nn.Conv2d(32, out_channels, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 假设输入x的尺寸为 [B, 4, H, W]
        B, C, H, W = x.shape

        # Encoder
        enc1 = self.encoder1(x)  # [B, 32, H, W]
        x = self.pool1(enc1)  # [B, 32, H/2, W/2]

        enc2 = self.encoder2(x)  # [B, 64, H/2, W/2]
        x = self.pool2(enc2)  # [B, 64, H/4, W/4]

        enc3 = self.encoder3(x)  # [B, 128, H/4, W/4]
        x = self.pool3(enc3)  # [B, 128, H/8, W/8] (180->90->45->22)

        # Transformer Bottleneck
        x_trans, (H_patch, W_patch) = self.patch_embed(x)  # [B, N, 256], N = (H/8 * W/8) / 4
        x_trans = self.transformer_blocks(x_trans)

        # Reshape back to feature map
        N = H_patch * W_patch
        if x_trans.size(1) != N:
            x_trans = x_trans[:, :N, :]

        x_trans = x_trans.transpose(1, 2).reshape(B, 256, H_patch, W_patch)  # [B, 256, H_patch, W_patch]

        # Decoder
        x = self.upsample3(x_trans)  # [B, 256, H/4, W/4]

        # 确保enc3与x尺寸匹配
        if enc3.size(2) != x.size(2) or enc3.size(3) != x.size(3):
            enc3_resized = F.interpolate(enc3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            enc3_resized = enc3

        x = torch.cat([x, enc3_resized], dim=1)  # [B, 256+128=384, H/4, W/4]
        x = self.decoder3(x)  # [B, 128, H/4, W/4]

        x = self.upsample2(x)  # [B, 128, H/2, W/2]

        # 确保enc2与x尺寸匹配
        if enc2.size(2) != x.size(2) or enc2.size(3) != x.size(3):
            enc2_resized = F.interpolate(enc2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            enc2_resized = enc2

        x = torch.cat([x, enc2_resized], dim=1)  # [B, 128+64=192, H/2, W/2]
        x = self.decoder2(x)  # [B, 64, H/2, W/2]

        x = self.upsample1(x)  # [B, 64, H, W]

        # 确保enc1与x尺寸匹配
        if enc1.size(2) != x.size(2) or enc1.size(3) != x.size(3):
            enc1_resized = F.interpolate(enc1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            enc1_resized = enc1

        x = torch.cat([x, enc1_resized], dim=1)  # [B, 64+32=96, H, W]
        x = self.decoder1(x)  # [B, 32, H, W]

        # 最终输出，确保与输入尺寸一致
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        out = self.final_conv(x)
        return out
#
#
# # 简化版本（如果上述版本仍有问题）
# class SimpleTransUNet(nn.Module):
#     def __init__(self, in_channels=4, out_channels=4):
#         super(SimpleTransUNet, self).__init__()
#
#         # 使用标准的UNet架构，只在bottleneck加入Transformer
#         self.encoder1 = self._block(in_channels, 32)
#         self.encoder2 = self._block(32, 64)
#         self.encoder3 = self._block(64, 128)
#
#         self.pool = nn.MaxPool2d(2)
#
#         # Bottleneck with simple attention
#         self.bottleneck = self._block(128, 256)
#         self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
#
#         self.decoder3 = self._block(256 + 128, 128)
#         self.decoder2 = self._block(128 + 64, 64)
#         self.decoder1 = self._block(64 + 32, 32)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.final_conv = nn.Conv2d(32, out_channels, 1)
#
#     def _block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         # Encoder
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool(enc1))
#         enc3 = self.encoder3(self.pool(enc2))
#
#         # Bottleneck with attention
#         bottleneck = self.bottleneck(self.pool(enc3))
#         B, C, H, W = bottleneck.shape
#         bottleneck_flat = bottleneck.flatten(2).transpose(1, 2)  # [B, N, C]
#         attn_out, _ = self.attention(bottleneck_flat, bottleneck_flat, bottleneck_flat)
#         bottleneck = (attn_out.transpose(1, 2).reshape(B, C, H, W) + bottleneck) * 0.5
#
#         # Decoder
#         dec3 = self.decoder3(torch.cat([self.upsample(bottleneck), enc3], dim=1))
#         dec2 = self.decoder2(torch.cat([self.upsample(dec3), enc2], dim=1))
#         dec1 = self.decoder1(torch.cat([self.upsample(dec2), enc1], dim=1))
#
#         return self.final_conv(dec1)