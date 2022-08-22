import torch
from torch import nn


class AttentionAggregator(nn.Module):
    """Aggregate features with computed attention value."""

    def __init__(self, in_features_size, inner_feature_size=128, out_feature_size=256):
        super().__init__()

        self.in_features_size = in_features_size  # size of flatten feature
        self.L = out_feature_size
        self.D = inner_feature_size
        
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

    def forward(self, x):
        x = x.view(-1, self.in_features_size)  # flatten feature，[N, C * H * W]
        x = self.fc1(x)  # [B, L]

        a = self.attention(x)  # attention value，[N, 1]
        a = torch.transpose(a, 1, 0)  # [1, N]
        a = torch.softmax(a, dim=1)

        m = torch.mm(a, x)  # [1, N] * [N, L] = [1, L]

        return m, a
