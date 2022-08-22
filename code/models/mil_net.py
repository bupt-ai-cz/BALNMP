import torch
from torch import nn
from .backbones.backbone_builder import BackboneBuilder
from .attention_aggregator import AttentionAggregator


class MILNetImageOnly(nn.Module):
    """Training with image only"""

    def __init__(self, num_classes, backbone_name):
        super().__init__()

        print('training with image only')
        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, bag_data, clinical_data=None):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        result = self.classifier(aggregated_feature)

        return result, attention


class MILNetWithClinicalData(nn.Module):
    """Training with image and clinical data"""

    def __init__(self, num_classes, backbone_name, clinical_data_size=5, expand_times=10):
        super().__init__()

        print('training with image and clinical data')
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times  # expanding clinical data to match image features in dimensions

        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L + self.clinical_data_size * self.expand_times, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.expand_times).float()], dim=-1)  # feature fusion
        result = self.classifier(fused_data)

        return result, attention