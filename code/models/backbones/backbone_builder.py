from torch import nn
import torchvision
import os
import timm
from huggingface_hub import hf_hub_download
import torch

BACKBONES = [
    "vgg16_bn",
    "uni2-h-freeze",
    "uni2-h"
]


class BackboneBuilder(nn.Module):
    """Build backbone (image feature extractor) with the last fc layer removed.
    
    Supported feature extractor:
        - vgg16_bn: VGG16 with batch normalization (baseline model used this with trainable weights)
        - uni2-h-freeze: UNI2-h with frozen weights (feature extractor only)
        - uni2-h: UNI2-h with trainable weights
    
    """

    def __init__(self, backbone_name):
        super().__init__()

        if backbone_name not in BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Choose from: {', '.join(BACKBONES)}"
            )
        
        self.backbone_name = backbone_name

        if backbone_name == "vgg16_bn":
            
            self.extractor, self.output_features_size = self.vgg(backbone_name)
        elif backbone_name == "uni2-h-freeze":
            self.extractor, self.output_features_size = self.uni2_h()
            # Freeze UNI2-h weights - use as feature extractor only
            self._freeze_backbone()
            print("UNI2-h backbone loaded and frozen (feature extractor mode)")
        elif backbone_name == "uni2-h":
            self.extractor, self.output_features_size = self.uni2_h()
            print("UNI2-h backbone loaded with trainable weights")
            
        else:
            raise NotImplementedError

    def forward(self, x):
        patch_features = self.extractor(x)

        return patch_features

    def vgg(self, backbone_name):
        
        try:
            complete_backbone = torchvision.models.__dict__[backbone_name](
                weights=torchvision.models.VGG16_BN_Weights.DEFAULT
            )
        except TypeError:

            complete_backbone = torchvision.models.__dict__[backbone_name](pretrained=True)
        
        output_features_size = 512 * 7 * 7
        # remove last layer
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
        
    def uni2_h(self):
        """Load UNI2-h model from Hugging Face (frozen for feature extraction)"""
        output_features_size = 1536  # Output: [batch, 1536]
        
        #C:\Users\aoara\OneDrive\Documents\repos\CS230\code\models\backbones
        local_dir = "/content/drive/MyDrive/cs230output/uni2-h/"
        #os.path.abspath("../code/models/backbones/uni2-h/")
        os.makedirs(local_dir, exist_ok=True)
        model_path = os.path.join(local_dir, "pytorch_model.bin")
    
        if not os.path.exists(model_path):  
            print(f'downloading UNI2-h model to {local_dir}')
            try:
                hf_hub_download(
                    repo_id="MahmoodLab/UNI2-h",
                    filename="pytorch_model.bin",
                    local_dir=local_dir,
                    force_download=False,
                    token= os.getenv("HF_TOKEN") ############## HF_TOKEN
                )
            except Exception as e:
                raise RuntimeError(
                    f" Failed to download UNI2-h model.\n"
                    f"Error: {str(e)}\n"
                )
        else:
            print(f"Using cached UNI2-h model from {local_dir}")
        
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,  # No classification head - feature extraction mode
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = timm.create_model(pretrained=False, **timm_kwargs).to(device)
        
        extractor.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True), 
            strict=True
        )
        print("UNI2-h weights loaded")
        
        return extractor, output_features_size


    def _freeze_backbone(self):
        """Freeze all parameters in the backbone for feature extraction only."""
        for param in self.extractor.parameters():
            param.requires_grad = False
        
    def unfreeze_backbone(self):
        """Unfreeze all parameters to allow fine-tuning."""
        for param in self.extractor.parameters():
            param.requires_grad = True
        print(f"{self.backbone_name} backbone unfrozen")
    
    def freeze_layers_ratio(self, ratio=0.5):
        """
        Freeze first layer*ratio of backbone
        Ex: ratio=0.3 freezes the first 30% of layers.
        """
        if not (0 < ratio < 1):
            raise ValueError("ratio must be between 0 and 1")

        all_params = list(self.extractor.parameters())
        total_layers = len(all_params)
        freeze_count = int(total_layers * ratio)

        for param in all_params[:freeze_count]:
            param.requires_grad = False
        for param in all_params[freeze_count:]:
            param.requires_grad = True

   

