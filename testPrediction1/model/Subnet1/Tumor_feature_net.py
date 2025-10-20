import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TumorFeatureExtractor(nn.Module):
    def __init__(self, m_length=32, device=None):
        super(TumorFeatureExtractor, self).__init__()
        self.m_length = m_length
        self.device = device

        # Use modern PyTorch weight loading API
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolution to accept single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy pretrained weights (average across the RGB channels)
        with torch.no_grad():
            pretrained_weights = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

        # Copy other ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, m_length)
        )

        # Initialize custom weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of custom layers."""
        for m in self.feature_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_nonzero_features(self, x):
        """Extract features from regions that are non-zero."""
        batch_size = x.shape[0]

        # Create a mask and check if there are any non-zero pixels
        mask = (x != 0).float()
        if torch.sum(mask) == 0:
            return torch.zeros(batch_size, self.m_length, device=x.device)

        # Use the full ResNet feature extraction pipeline
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Feature projection
        features = self.feature_projection(x)

        return features

    def forward(self, x):
        if len(x.shape) == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2).contiguous()

        # Ensure input is on the correct device
        if self.device is not None:
            x = x.to(self.device)

        return self.extract_nonzero_features(x)


class NonZeroFeatureExtractor(nn.Module):
    """Module specifically for extracting features from non-zero pixel regions."""

    def __init__(self, m_length):
        super(NonZeroFeatureExtractor, self).__init__()
        self.m_length = m_length

        # Local feature extraction network
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

        # Shape feature extractor
        self.shape_feature_extractor = ShapeFeatureExtractor(64)

        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, m_length)
        )

    def forward(self, x, mask):
        batch_size = x.shape[0]

        # Extract local texture features
        local_features = self.local_feature_net(x)
        local_features = local_features.view(batch_size, -1)

        # Extract shape features
        shape_features = self.shape_feature_extractor(mask)

        # Fuse features
        combined_features = torch.cat([local_features, shape_features], dim=1)
        final_features = self.feature_fusion(combined_features)

        return final_features


class ShapeFeatureExtractor(nn.Module):
    """Module to extract tumor shape features."""

    def __init__(self, output_dim):
        super(ShapeFeatureExtractor, self).__init__()
        self.output_dim = output_dim

        # Shape feature extraction network
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

        # Global shape statistics
        self.global_stats = GlobalShapeStatistics(16)

        self.feature_projection = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, mask):
        batch_size = mask.shape[0]


        if len(mask.shape) == 4 and mask.shape[1] == 1:
            mask_2d = mask
        else:
            mask_2d = mask.unsqueeze(1)

        # Extract convolutional features
        conv_features = self.shape_net(mask_2d)
        conv_features = conv_features.view(batch_size, -1)

        # Extract global statistical features
        global_features = self.global_stats(mask_2d)

        # Combine features
        combined = torch.cat([conv_features, global_features], dim=1)
        output = self.feature_projection(combined)

        return output


class GlobalShapeStatistics(nn.Module):
    """Compute global shape statistical features for the tumor - optimized version."""

    def __init__(self, output_dim):
        super(GlobalShapeStatistics, self).__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(6, output_dim)  # reduced feature dimensionality

    def forward(self, mask):
        batch_size = mask.shape[0]
        features = []

        for i in range(batch_size):
            single_mask = mask[i, 0]
            nonzero_coords = torch.nonzero(single_mask)

            if len(nonzero_coords) == 0:
                stats = torch.zeros(6, device=mask.device)
            else:
                # Compute basic shape statistics
                coords_float = nonzero_coords.float()
                center = torch.mean(coords_float, dim=0)
                max_coords = torch.max(coords_float, dim=0)[0]
                min_coords = torch.min(coords_float, dim=0)[0]

                height = max_coords[0] - min_coords[0]
                width = max_coords[1] - min_coords[1]
                area = len(nonzero_coords)

                # Normalized features
                h, w = single_mask.shape
                stats = torch.stack([
                    center[0] / h,  # normalized y-center
                    center[1] / w,  # normalized x-center
                    height / h,     # normalized height
                    width / w,      # normalized width
                    area / (h * w), # area ratio
                    (height * width) / (h * w)  # bounding-box ratio
                ])

            features.append(stats)

        features = torch.stack(features)
        return self.projection(features)


# Specific three glioma-modality networks
class TumorCoreNet(TumorFeatureExtractor):
    """Tumor core-region feature extraction network."""

    def __init__(self, m_length=32, device=None):
        super(TumorCoreNet, self).__init__(m_length, device)
        print("Initializing TumorCoreNet")


class EnhancingTumorNet(TumorFeatureExtractor):
    """Enhancing tumor-region feature extraction network."""

    def __init__(self, m_length=32, device=None):
        super(EnhancingTumorNet, self).__init__(m_length, device)
        print("Initializing EnhancingTumorNet")


class WholeTumorNet(TumorFeatureExtractor):
    """Whole-tumor region feature extraction network."""

    def __init__(self, m_length=32, device=None):
        super(WholeTumorNet, self).__init__(m_length, device)
        print("Initializing WholeTumorNet")