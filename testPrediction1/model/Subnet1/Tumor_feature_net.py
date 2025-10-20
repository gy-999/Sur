import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TumorFeatureExtractor(nn.Module):
    def __init__(self, m_length=32, device=None):
        super(TumorFeatureExtractor, self).__init__()
        self.m_length = m_length
        self.device = device

        # 使用现代PyTorch的权重加载方式
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 修改第一层卷积以适应单通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 复制预训练权重（对RGB通道取平均）
        with torch.no_grad():
            pretrained_weights = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

        # 复制ResNet的其他层
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, m_length)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化自定义层的权重"""
        for m in self.feature_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_nonzero_features(self, x):
        """提取非零像素区域的特征"""
        batch_size = x.shape[0]

        # 创建mask并检查是否有非零像素
        mask = (x != 0).float()
        if torch.sum(mask) == 0:
            return torch.zeros(batch_size, self.m_length, device=x.device)

        # 使用完整的ResNet特征提取流程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # 特征投影
        features = self.feature_projection(x)

        return features

    def forward(self, x):
        # 处理输入维度 [B, 240, 240, 1] -> [B, 1, 240, 240]
        if len(x.shape) == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2).contiguous()

        # 确保输入在正确设备上
        if self.device is not None:
            x = x.to(self.device)

        return self.extract_nonzero_features(x)


class NonZeroFeatureExtractor(nn.Module):
    """专门用于提取非零像素区域特征的模块"""

    def __init__(self, m_length):
        super(NonZeroFeatureExtractor, self).__init__()
        self.m_length = m_length

        # 局部特征提取网络
        self.local_feature_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 形状特征提取器
        self.shape_feature_extractor = ShapeFeatureExtractor(64)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, m_length)
        )

    def forward(self, x, mask):
        batch_size = x.shape[0]

        # 提取局部纹理特征
        local_features = self.local_feature_net(x)
        local_features = local_features.view(batch_size, -1)

        # 提取形状特征
        shape_features = self.shape_feature_extractor(mask)

        # 融合特征
        combined_features = torch.cat([local_features, shape_features], dim=1)
        final_features = self.feature_fusion(combined_features)

        return final_features


class ShapeFeatureExtractor(nn.Module):
    """提取肿瘤形状特征的模块"""

    def __init__(self, output_dim):
        super(ShapeFeatureExtractor, self).__init__()
        self.output_dim = output_dim

        # 形状特征提取网络
        self.shape_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 全局形状统计
        self.global_stats = GlobalShapeStatistics(16)

        self.feature_projection = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, mask):
        batch_size = mask.shape[0]

        # 确保mask是2D的 [B, 1, H, W]
        if len(mask.shape) == 4 and mask.shape[1] == 1:
            mask_2d = mask
        else:
            mask_2d = mask.unsqueeze(1)

        # 提取卷积特征
        conv_features = self.shape_net(mask_2d)
        conv_features = conv_features.view(batch_size, -1)

        # 提取全局统计特征
        global_features = self.global_stats(mask_2d)

        # 合并特征
        combined = torch.cat([conv_features, global_features], dim=1)
        output = self.feature_projection(combined)

        return output


class GlobalShapeStatistics(nn.Module):
    """计算肿瘤的全局形状统计特征 - 优化版本"""

    def __init__(self, output_dim):
        super(GlobalShapeStatistics, self).__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(6, output_dim)  # 减少特征维度

    def forward(self, mask):
        batch_size = mask.shape[0]
        features = []

        for i in range(batch_size):
            single_mask = mask[i, 0]  # [H, W]
            nonzero_coords = torch.nonzero(single_mask)

            if len(nonzero_coords) == 0:
                stats = torch.zeros(6, device=mask.device)
            else:
                # 计算基本形状特征
                coords_float = nonzero_coords.float()
                center = torch.mean(coords_float, dim=0)
                max_coords = torch.max(coords_float, dim=0)[0]
                min_coords = torch.min(coords_float, dim=0)[0]

                height = max_coords[0] - min_coords[0]
                width = max_coords[1] - min_coords[1]
                area = len(nonzero_coords)

                # 归一化特征
                h, w = single_mask.shape
                stats = torch.stack([
                    center[0] / h,  # 归一化y中心
                    center[1] / w,  # 归一化x中心
                    height / h,  # 归一化高度
                    width / w,  # 归一化宽度
                    area / (h * w),  # 面积比例
                    (height * width) / (h * w)  # 边界框比例
                ])

            features.append(stats)

        features = torch.stack(features)
        return self.projection(features)


# 具体的三个胶质瘤模态网络
class TumorCoreNet(TumorFeatureExtractor):
    """肿瘤核心区域特征提取网络"""

    def __init__(self, m_length=32, device=None):
        super(TumorCoreNet, self).__init__(m_length, device)
        print("初始化 TumorCoreNet")


class EnhancingTumorNet(TumorFeatureExtractor):
    """增强肿瘤区域特征提取网络"""

    def __init__(self, m_length=32, device=None):
        super(EnhancingTumorNet, self).__init__(m_length, device)
        print("初始化 EnhancingTumorNet")


class WholeTumorNet(TumorFeatureExtractor):
    """全肿瘤区域特征提取网络"""

    def __init__(self, m_length=32, device=None):
        super(WholeTumorNet, self).__init__(m_length, device)
        print("初始化 WholeTumorNet")

