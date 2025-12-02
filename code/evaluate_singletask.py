import torch
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from mil_net import Singletask_MILNET
from backbone_builder import BACKBONES
from dataset_loader import BreastDataset

def get_test_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    
    # Dataset args
    parser.add_argument("--test_json_path", default="./dataset/json/updated_test-type-0.json")
    parser.add_argument("--data_dir_path", required=True, help="Path to patches")
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")  
    parser.add_argument("--preloading", action="store_true")

    # Model args (Must match the trained model settings)
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")
    parser.add_argument("--dropout", type=float, default=0.2)
   
    
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_combined.pth or best_meta.pth")
    parser.add_argument("--num_workers", type=int, default=8)

    return parser.parse_args()

def calculate_extended_metrics(labels, probs, preds):
    """
    Calculates detailed metrics for binary classification.
    Assumes labels are 0 and 1.
    """
    metrics = {}
    
    # Basic Metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['auc'] = roc_auc_score(labels, probs)
    metrics['f1'] = f1_score(labels, preds)
    
    # Confusion Matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # Extended Clinical Metrics
    # Sensitivity (Recall) = TP / (TP + FN)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['recall'] = metrics['sensitivity'] # Recall is same as Sensitivity
    
    # Specificity = TN / (TN + FP)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Positive Predictive Value (Precision) = TP / (TP + FP)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['precision'] = metrics['ppv'] # Precision is same as PPV
    
    # Negative Predictive Value = TN / (TN + FN)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return metrics

def test(model, dataloader, args):
    model.eval()
    
    # Arrays to store results
    meta_preds, meta_labels, meta_probs = [], [], []
  
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            bag_tensor = data["bag_tensor"].cuda()
            clinical_data = data["clinical_data"].cuda()
            metastasis_label = data["metastasis_label"].cuda()

            
           
            meta_logits, _ = model(bag_tensor, clinical_data)
            
            # --- Process Metastasis (Binary) ---
            meta_prob = torch.softmax(meta_logits, dim=1)[:, 1] # Probability of class 1
            meta_pred = torch.argmax(meta_logits, dim=1)
            
            meta_probs.extend(meta_prob.cpu().numpy())
            meta_preds.extend(meta_pred.cpu().numpy())
            meta_labels.extend(metastasis_label.cpu().numpy())
            
        

    # Convert to numpy
    meta_probs = np.array(meta_probs)
    meta_preds = np.array(meta_preds)
    meta_labels = np.array(meta_labels)
    
    
    # --- Compute Metrics ---
    
    print("\n" + "="*30)
    print("  METASTASIS (Binary) RESULTS  ")
    print("="*30)
    
    meta_metrics = calculate_extended_metrics(meta_labels, meta_probs, meta_preds)
    
    for k, v in meta_metrics.items():
        print(f"{k.upper():<15}: {v:.4f}")

 
    

if __name__ == "__main__":
    args = get_test_args()
    
    # 1. Load Dataset
    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # 2. Initialize Model
    print(f"Initializing model with backbone: {args.backbone}")
 
    model = Singletask_MILNET(backbone_name=args.backbone, dropout=args.dropout)
    
    model = model.cuda()
    
    # 3. Load Checkpoint
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")
        
    # 4. Run Test
    test(model, test_loader, args)