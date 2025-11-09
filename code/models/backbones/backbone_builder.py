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
    """Build backbone with the last fc layer removed"""

    def __init__(self, backbone_name):
        super().__init__()

        assert backbone_name in BACKBONES

        if backbone_name.startswith("vgg"):
            complete_backbone = torchvision.models.__dict__[backbone_name](pretrained=True)
            assert backbone_name in [ "vgg16_bn"]
            self.extractor, self.output_features_size = self.vgg(complete_backbone)
        elif backbone_name == "uni2-h-freeze":
            self.extractor, self.output_features_size = self.uni2_h()
            # Freeze UNI2-h weights - use as feature extractor only
            for param in self.extractor.parameters():
                param.requires_grad = False
            print("UNI2-h backbone loaded and frozen (feature extractor mode)")
        elif backbone_name == "uni2-h":
            self.extractor, self.output_features_size = self.uni2_h()
            print("not freezing layers in uni2-h")
            
        else:
            raise NotImplementedError

    def forward(self, x):
        patch_features = self.extractor(x)

        return patch_features

    def vgg(self, complete_backbone):
        output_features_size = 512 * 7 * 7
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
        
    def uni2_h(self):
        """Load UNI2-h model from Hugging Face (frozen for feature extraction)"""
        output_features_size = 1536 * 1 * 1  # Output: [batch, 1536]
        
        #C:\Users\aoara\OneDrive\Documents\repos\CS230\code\models\backbones
        local_dir = "/content/drive/MyDrive/cs230output/uni2-h/"
        #os.path.abspath("../code/models/backbones/uni2-h/")
        os.makedirs(local_dir, exist_ok=True)
        model_path = os.path.join(local_dir, "pytorch_model.bin")
    
        if not os.path.exists(model_path):  
            try:
                hf_hub_download(
                    repo_id="MahmoodLab/UNI2-h",
                    filename="pytorch_model.bin",
                    local_dir=local_dir,
                    force_download=False,
                    token=True ############## HF_TOKEN
                )
                print("✓ Download complete!")
            except Exception as e:
                raise RuntimeError(
                    f" Failed to download UNI2-h model.\n"
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
        
        #print("Loading UNI2-h model")
        extractor = timm.create_model(pretrained=False, **timm_kwargs)
        
        extractor.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True), 
            strict=True
        )
        print("UNI2-h weights loaded")
        
        return extractor, output_features_size

   

