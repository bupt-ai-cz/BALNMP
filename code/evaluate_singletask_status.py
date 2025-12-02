import torch
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from mil_net import Singletask_MILNET_status
from backbone_builder import BACKBONES
from dataset_loader import BreastDataset

def get_test_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Status Classification")
    
    # Dataset args
    parser.add_argument("--test_json_path", default="./dataset/json/updated_test-type-0.json")
    parser.add_argument("--data_dir_path", required=True, help="Path to patches")
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")  
    parser.add_argument("--preloading", action="store_true")

    # Model args (Must match the trained model settings)
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")
    parser.add_argument("--dropout", type=float, default=0.2)
   
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_status.pth")
    parser.add_argument("--num_workers", type=int, default=8)

    return parser.parse_args()

def calculate_multiclass_metrics(labels, probs, preds):
    """
    Calculates detailed metrics for multi-class classification (3 classes).
    Classes: N0 (0), N+(1-2) (1), N+(>2) (2)
    """
    metrics = {}
    
    # Basic Metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # Macro-averaged metrics
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    metrics['f1_class_0'] = f1_per_class[0]  # N0
    metrics['f1_class_1'] = f1_per_class[1]  # N+(1-2)
    metrics['f1_class_2'] = f1_per_class[2]  # N+(>2)
    
    # AUC (one-vs-rest)
    try:
        metrics['auc_macro'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        metrics['auc_weighted'] = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except ValueError:
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm
    
    # Per-class metrics from confusion matrix
    for i in range(3):
        # True Positives for class i
        tp = cm[i, i]
        # False Positives for class i (predicted as i but not actually i)
        fp = cm[:, i].sum() - tp
        # False Negatives for class i (actually i but not predicted as i)
        fn = cm[i, :].sum() - tp
        # True Negatives for class i
        tn = cm.sum() - tp - fp - fn
        
        # Sensitivity (Recall) for class i
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'sensitivity_class_{i}'] = sensitivity
        
        # Specificity for class i
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'specificity_class_{i}'] = specificity
        
        # Precision (PPV) for class i
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f'precision_class_{i}'] = precision
        
        # NPV for class i
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics[f'npv_class_{i}'] = npv
    
    return metrics

def test(model, dataloader, args):
    model.eval()
    
    # Arrays to store results
    status_preds, status_labels, status_probs = [], [], []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            bag_tensor = data["bag_tensor"].cuda()
            clinical_data = data["clinical_data"].cuda()
            status_label = data["status_label"].cuda()
            
            # Forward pass
            status_logits, _ = model(bag_tensor, clinical_data)
            
            # --- Process Status (Multi-class: 3 classes) ---
            status_prob = torch.softmax(status_logits, dim=1)  # Probabilities for all classes
            status_pred = torch.argmax(status_logits, dim=1)
            
            status_probs.extend(status_prob.cpu().numpy())
            status_preds.extend(status_pred.cpu().numpy())
            status_labels.extend(status_label.cpu().numpy())

    # Convert to numpy
    status_probs = np.array(status_probs)
    status_preds = np.array(status_preds)
    status_labels = np.array(status_labels)
    
    # --- Compute Metrics ---
    print("\n" + "="*40)
    print("  STATUS (Multi-class: 3 classes) RESULTS  ")
    print("="*40)
    print("Classes: 0=N0, 1=N+(1-2), 2=N+(>2)")
    print("="*40)
    
    status_metrics = calculate_multiclass_metrics(status_labels, status_probs, status_preds)
    
    # Print overall metrics
    print("\n--- Overall Metrics ---")
    print(f"{'ACCURACY':<20}: {status_metrics['accuracy']:.4f}")
    print(f"{'AUC_MACRO':<20}: {status_metrics['auc_macro']:.4f}")
    print(f"{'AUC_WEIGHTED':<20}: {status_metrics['auc_weighted']:.4f}")
    print(f"{'F1_MACRO':<20}: {status_metrics['f1_macro']:.4f}")
    print(f"{'F1_WEIGHTED':<20}: {status_metrics['f1_weighted']:.4f}")
    
    # Print per-class F1 scores
    print("\n--- Per-Class F1 Scores ---")
    print(f"{'F1_CLASS_0 (N0)':<20}: {status_metrics['f1_class_0']:.4f}")
    print(f"{'F1_CLASS_1 (N+(1-2))':<20}: {status_metrics['f1_class_1']:.4f}")
    print(f"{'F1_CLASS_2 (N+(>2))':<20}: {status_metrics['f1_class_2']:.4f}")
    
    # Print per-class detailed metrics
    print("\n--- Per-Class Detailed Metrics ---")
    for i in range(3):
        class_name = ["N0", "N+(1-2)", "N+(>2)"][i]
        print(f"\nClass {i} ({class_name}):")
        print(f"  {'Sensitivity':<15}: {status_metrics[f'sensitivity_class_{i}']:.4f}")
        print(f"  {'Specificity':<15}: {status_metrics[f'specificity_class_{i}']:.4f}")
        print(f"  {'Precision':<15}: {status_metrics[f'precision_class_{i}']:.4f}")
        print(f"  {'NPV':<15}: {status_metrics[f'npv_class_{i}']:.4f}")
    
    # Print confusion matrix
    print("\n--- Confusion Matrix ---")
    print("Rows: True Labels, Columns: Predicted Labels")
    print(status_metrics['confusion_matrix'])
    
    # Print classification report
    print("\n--- Classification Report ---")
    target_names = ['N0', 'N+(1-2)', 'N+(>2)']
    print(classification_report(status_labels, status_preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    args = get_test_args()
    
    # 1. Load Dataset
    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # 2. Initialize Model
    print(f"Initializing model with backbone: {args.backbone}")
    model = Singletask_MILNET_status(backbone_name=args.backbone, dropout=args.dropout)
    model = model.cuda()
    
    # 3. Load Checkpoint
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")
        
    # 4. Run Test
    test(model, test_loader, args)
