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

    def __init__(self, backbone_name, clinical_data_size=5, expand_times=10, dropout = 0.2):
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
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # num_classes =  3 # there are three classes - N0 --> 0 ;  N+(1-2) --> 1 ;  N+(>2)  --> 2
        self.status_classifier = nn.Sequential(
            nn.Linear(shared_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    
    
    
class Multitask_MILNET_image_only(nn.Module):
    """Multi-task MIL model predicting both metastasis and status. Using Image data only.
      Uses separate classification heads WITHOUT shared layers after attention module.
    """

    def __init__(self, backbone_name, dropout = 0.2):
        super().__init__()

        print('training with image data only')
        
        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(self.image_feature_extractor.output_features_size, 1)  # inner_feature_size=1
        shared_feature_size = self.attention_aggregator.L 
        
        # using num_classes = 2
        self.metastasis_classifier = nn.Sequential(
            nn.Linear(shared_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # num_classes =  3 # there are three classes - N0 --> 0 ;  N+(1-2) --> 1 ;  N+(>2)  --> 2
        self.status_classifier = nn.Sequential(
            nn.Linear(shared_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        ) 
        

    def forward(self, bag_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        metastasis_result = self.metastasis_classifier(aggregated_feature)
        status_result = self.status_classifier(aggregated_feature)

        return metastasis_result, status_result, attention


class Singletask_MILNET(nn.Module):
    """Single-task MIL model predicting both metastasis ONLY. Use as a comparison for Multitask_MILNET
      Uses separate classification heads WITHOUT shared layers after attention module.
    """

    def __init__(self, backbone_name, clinical_data_size=5, expand_times=10, dropout = 0.2):
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
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        
    def forward(self, bag_data, clinical_data):
        bag_data = bag_data.squeeze(0)  # [1 (batch size), N, C, H, W] --> [N, C, H, W], remove the batch dimension
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(patch_features)
        # adding clinical data features, expand by 10 times
        fused_data = torch.cat([aggregated_feature, clinical_data.repeat(1, self.expand_times).float()], dim=-1)  # feature fusion
        metastasis_result = self.metastasis_classifier(fused_data)
        

        return metastasis_result, attention


#  baseline model attributed to: 
#     @article{xu2021predicting,
#   title={Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides},
#   author={Xu, Feng and Zhu, Chuang and Tang, Wenqi and Wang, Ying and Zhang, Yu and Li, Jie and Jiang, Hongchuan and Shi, Zhongyue and Liu, Jun and Jin, Mulan},
#   journal={Frontiers in oncology},
#   volume={11},
#   pages={759007},
#   year={2021},
#   publisher={Frontiers Media SA}
# }
# github link: https://github.com/bupt-ai-cz/BALNMP
