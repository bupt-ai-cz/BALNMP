# -*- coding: utf-8 -*
import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class AttentionAggregator(nn.Module):

    def __init__(self, in_features_size, inner_feature_size=128, out_feature_size=256, is_dropout=True):
        super().__init__()

        self.in_features_size = in_features_size
        self.L = out_feature_size
        self.D = inner_feature_size

        if is_dropout:
            self.fc1 = nn.Sequential(
                nn.Linear(self.in_features_size, self.L),
                nn.Dropout(),
                nn.ReLU()
            )
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Dropout(),
                nn.Tanh(),
                nn.Linear(self.D, 1),
                nn.Dropout()
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(self.in_features_size, self.L),
                nn.ReLU()
            )
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
                nn.Linear(self.D, 1),
            )

    def forward(self, x):
        x = x.view(-1, self.in_features_size)  # [bag_size, channel * height * width]
        x = self.fc1(x)  # [bag_size, L]

        a = self.attention(x)  # [bag_size, 1]
        a = torch.transpose(a, 1, 0)  # [1, bag_size]
        a = F.softmax(a, dim=1)

        m = torch.mm(a, x)  # [1, bag_size] * [bag_size, L] = [1, L]

        return m, a


class MILMainNetWithClinicalData(nn.Module):

    def __init__(self, num_classes=2, clinical_data_size=5):
        super().__init__()
        self.build_image_feature_extractor()

        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)

        self.clinical_data_size = clinical_data_size
        self.scale_rate = 10
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L + self.clinical_data_size * self.scale_rate, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))

    def build_image_feature_extractor(self):
        self.image_feature_extractor = nn.Sequential(*(list(torchvision.models.vgg16_bn(pretrained=True).children())[:-1]))
        self.image_feature_extractor.output_features_size = 512 * 7 * 7

    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.scale_rate).float()], dim=-1)
        result = self.classifier(fused_data)

        return result, attention
