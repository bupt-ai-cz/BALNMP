import torch
from torch import nn
from backbone_builder import BackboneBuilder
from attention_aggregator import AttentionAggregator

class MILNetWithClinicalData(nn.Module):
    """Baseline MIL model with image and clinical data fusion."""

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
    
 ## multi-task classifier branches w/o an additional shared layer
class Multitask_MILNET(nn.Module):
    """Multi-task MIL model predicting both metastasis and status.
      Uses separate classification heads WITHOUT shared layers after attention module.
    """

    def __init__(self, backbone_name, clinical_data_size=5, expand_times=10):
        super().__init__()

        print('training with image and clinical data')
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times  # expanding clinical data to match image features in dimensions

        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        shared_feature_size = self.attention_aggregator.L + self.clinical_data_size * self.expand_times
        
        # using num_classes = 2
        self.metastasis_classifier = nn.Sequential(
            nn.Linear(shared_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        # num_classes =  3 # there are three classes - N0 --> 0 ;  N+(1-2) --> 1 ;  N+(>2)  --> 2
        self.status_classifier = nn.Sequential(
            nn.Linear(shared_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        ) 
        

    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        # adding clinical data features, expand by 10 times
        fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.expand_times).float()], dim=-1)  # feature fusion
        metastasis_result = self.metastasis_classifier(fused_data)
        status_result = self.status_classifier(fused_data)

        return metastasis_result, status_result, attention
  
class Multitask_MILNET_large(nn.Module):
    """Multi-task MIL model WITH shared layer after attention module"""

    def __init__(self, backbone_name, clinical_data_size=5, expand_times=10):
        super().__init__()

        print('training with image and clinical data')
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times  # expanding clinical data to match image features in dimensions

        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        shared_feature_size = self.attention_aggregator.L + self.clinical_data_size * self.expand_times
        
        # add a shared layer
        self.shared_layer = nn.Sequential(
            nn.Linear(shared_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # using num_classes = 2
        self.metastasis_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        # num_classes =  3 # there are three classes - N0 --> 0 ;  N+(1-2) --> 1 ;  N+(>2)  --> 2
        self.status_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        ) 
        

    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        # adding clinical data features, expand by 10 times
        fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.expand_times).float()], dim=-1)  # feature fusion
        shared_features = self.shared_layer(fused_data)
        
        metastasis_result = self.metastasis_classifier(shared_features)
        status_result = self.status_classifier(shared_features)

        return metastasis_result, status_result, attention
        
        
        