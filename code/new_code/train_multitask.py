import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from mil_net import Multitask_MILNET
from backbone_builder import BACKBONES
from dataset_loader import BreastDataset
import random
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm


def parser_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_json_path", default="./dataset/json/updated_train-type-0.json")
    parser.add_argument("--val_json_path", default="./dataset/json/updated_val-type-0.json")
    parser.add_argument("--test_json_path", default="./dataset/json/updated_test-type-0.json")
    parser.add_argument("--data_dir_path", required= True) # path to .../ patches
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")  
    parser.add_argument("--preloading", action="store_true")

    # model
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")

    # optimizer
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # scheduler
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use learning rate scheduler (from baseline implementation)")

    # multi-task loss weights
    parser.add_argument("--metastasis_loss_weight", type=float, default=1.0)
    parser.add_argument("--status_loss_weight", type=float, default=1.0)

    # output
    parser.add_argument("--log_dir_path", required=True)
    parser.add_argument("--log_name", required=True,  help="Experiment name")
    parser.add_argument("--save_epoch_interval", type=int, default=10)

    # other
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    return args


def init_output_directory(log_dir_path, log_name):
    tensorboard_path = os.path.join(log_dir_path, log_name, "tensorboard")
    checkpoint_path = os.path.join(log_dir_path, log_name, "checkpoint")
    xlsx_path = os.path.join(log_dir_path, log_name, "xlsx")

    for path in [tensorboard_path, checkpoint_path, xlsx_path]:
        os.makedirs(path, exist_ok=True)
        print(f"init path {path}")

    return tensorboard_path, checkpoint_path, xlsx_path


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_optimizer(args, model):
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError
    

def init_dataloader(args):
    train_dataset = BreastDataset(args.train_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_dataset = BreastDataset(args.val_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


def compute_metrics(labels, probs, preds, task_type='binary'):
    """
    Compute evaluation metrics.
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        preds: Predicted classes
        task_type: 'binary' or 'multiclass'
    
    Returns:
        dict: keys - f1, auc, accuracy
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # F1 Score
    if task_type == 'binary':
        metrics['f1'] = f1_score(labels, preds, average='binary', zero_division=0)
    else:
        metrics['f1'] = f1_score(labels, preds, average='macro', zero_division=0)
       # metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # AUC
    try:
        if task_type == 'binary':
            metrics['auc'] = roc_auc_score(labels, probs)
        else:
            metrics['auc'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        metrics['auc'] = 0.0
    
    # metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    
    return metrics

def train_epoch(model, dataloader, optimizer, metastasis_criterion, status_criterion, args, writer, epoch):
    model.train()
    
    metastasis_losses = []
    status_losses = []
    total_losses = []
    
    metastasis_preds = []
    metastasis_labels = []
    metastasis_probs = []
    
    status_preds = []
    status_labels = []
    status_probs = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [TRAIN]", ncols=120)
    for batch_idx, data in pbar:
        bag_tensor = data["bag_tensor"].cuda()
        clinical_data = data["clinical_data"].cuda()
        metastasis_label = data["metastasis_label"].cuda()
        status_label = data["status_label"].cuda()
        
        # Forward pass
        metastasis_logits, status_logits, _ = model(bag_tensor, clinical_data)
        
        # Calculate losses
        metastasis_loss = metastasis_criterion(metastasis_logits, metastasis_label)
        status_loss = status_criterion(status_logits, status_label)
        
        # Weighted total loss
        total_loss = (args.metastasis_loss_weight * metastasis_loss + 
                     args.status_loss_weight * status_loss)
        
        # Backward pass
        optimizer.zero_grad() # clear gradient
        total_loss.backward() # calculate gradient
        
        #  if args.gradient_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        
        optimizer.step() # update gradient
        
        # Record losses
        metastasis_losses.append(metastasis_loss.item())
        status_losses.append(status_loss.item())
        total_losses.append(total_loss.item())
        
        # Get predictions
        with torch.no_grad():
            metastasis_prob = torch.softmax(metastasis_logits, dim=1)[:, 1]
            metastasis_pred = torch.argmax(metastasis_logits, dim=1)
            
            status_prob = torch.softmax(status_logits, dim=1)
            status_pred = torch.argmax(status_logits, dim=1)
        
        # Store
        metastasis_probs.append(metastasis_prob.cpu().detach().numpy())
        metastasis_preds.append(metastasis_pred.cpu().detach().numpy())
        metastasis_labels.append(metastasis_label.cpu().detach().numpy())
        
        status_probs.append(status_prob.cpu().detach().numpy())
        status_preds.append(status_pred.cpu().detach().numpy())
        status_labels.append(status_label.cpu().detach().numpy())
        
        pbar.set_postfix({
            'meta_loss': f'{metastasis_loss.item():.4f}',
            'status_loss': f'{status_loss.item():.4f}',
            'total_loss': f'{total_loss.item():.4f}'
        })
    
    # concat into single numpy arrays
    metastasis_probs = np.concatenate(metastasis_probs)
    metastasis_preds = np.concatenate(metastasis_preds)
    metastasis_labels = np.concatenate(metastasis_labels)
    
    status_probs = np.concatenate(status_probs, axis=0)
    status_preds = np.concatenate(status_preds)
    status_labels = np.concatenate(status_labels)
    
    # Metastasis metrics
   # Compute metrics
    metastasis_metrics = compute_metrics(
        metastasis_labels, metastasis_probs, metastasis_preds, task_type='binary'
    )
    status_metrics = compute_metrics(
        status_labels, status_probs, status_preds, task_type='multiclass'
    )
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('train/metastasis_loss', np.mean(metastasis_losses), epoch)
        writer.add_scalar('train/status_loss', np.mean(status_losses), epoch)
        writer.add_scalar('train/total_loss', np.mean(total_losses), epoch)
        writer.add_scalar('train/metastasis_auc', metastasis_metrics['auc'], epoch)
        writer.add_scalar('train/metastasis_acc', metastasis_metrics['accuracy'], epoch)
        writer.add_scalar('train/status_auc', status_metrics['auc'], epoch)
        writer.add_scalar('train/status_acc', status_metrics['accuracy'], epoch)
       
    
    print(f"[TRAIN] Epoch {epoch} | "
          f"Meta Loss: {np.mean(metastasis_losses):.4f} | Meta AUC: {metastasis_metrics['auc']:.4f} | Meta ACC: {metastasis_metrics['accuracy']:.4f} | "
          f"Status Loss: {np.mean(status_losses):.4f} | Status AUC: {status_metrics['auc']:.4f} | Status ACC: {status_metrics['accuracy']:.4f}")
    
    return  metastasis_metrics['auc'], status_metrics['auc']


def evaluate(model, dataloader, metastasis_criterion, status_criterion, args, writer, epoch, phase='val'):
    model.eval()
    
    metastasis_losses = []
    status_losses = []
    
    metastasis_preds = []
    metastasis_labels = []
    metastasis_probs = []
    
    status_preds = []
    status_labels = []
    status_probs = []
    
    with torch.no_grad():
        for data in dataloader:
            bag_tensor = data["bag_tensor"].cuda()
            clinical_data = data["clinical_data"].cuda()
            metastasis_label = data["metastasis_label"].cuda()
            status_label = data["status_label"].cuda()
            
            # Forward pass
            metastasis_logits, status_logits, _ = model(bag_tensor, clinical_data)
            
            # Calculate losses
            metastasis_loss = metastasis_criterion(metastasis_logits, metastasis_label)
            status_loss = status_criterion(status_logits, status_label)
            
            metastasis_losses.append(metastasis_loss.item())
            status_losses.append(status_loss.item())
            
            # Get predictions
            # dim 0 - batch dimension; dim 1 - class unnormalized logits; [:, 1] get probability of positive label (metasasis)
            metastasis_prob = torch.softmax(metastasis_logits, dim=1)[:, 1]
            metastasis_pred = torch.argmax(metastasis_logits, dim=1)
            # status has three classes 
            status_prob = torch.softmax(status_logits, dim=1)
            status_pred = torch.argmax(status_logits, dim=1)
            
            # Store 
            metastasis_probs.append(metastasis_prob.cpu().numpy())
            metastasis_preds.append(metastasis_pred.cpu().numpy())
            metastasis_labels.append(metastasis_label.cpu().numpy())
            
            status_probs.append(status_prob.cpu().numpy())
            status_preds.append(status_pred.cpu().numpy())
            status_labels.append(status_label.cpu().numpy())
    
    # concat into single numpy arrays
    metastasis_probs = np.concatenate(metastasis_probs)
    metastasis_preds = np.concatenate(metastasis_preds)
    metastasis_labels = np.concatenate(metastasis_labels)
    
    status_probs = np.concatenate(status_probs, axis=0)
    status_preds = np.concatenate(status_preds)
    status_labels = np.concatenate(status_labels)
    
    # Metrics
    metastasis_metrics = compute_metrics(
        metastasis_labels, metastasis_probs, metastasis_preds, task_type='binary'
    )
    status_metrics = compute_metrics(
        status_labels, status_probs, status_preds, task_type='multiclass'
    )
    
    # Log to tensorboard
    if writer:
        writer.add_scalar(f'{phase}/metastasis_loss', np.mean(metastasis_losses), epoch)
        writer.add_scalar(f'{phase}/status_loss', np.mean(status_losses), epoch)
        writer.add_scalar(f'{phase}/metastasis_auc', metastasis_metrics['auc'], epoch)
        writer.add_scalar(f'{phase}/metastasis_acc', metastasis_metrics['accuracy'], epoch)
        writer.add_scalar(f'{phase}/status_auc', status_metrics['auc'], epoch)
        writer.add_scalar(f'{phase}/status_acc', status_metrics['accuracy'], epoch)
    
    print(f"[{phase.upper()}] Epoch {epoch} | "
          f"Meta Loss: {np.mean(metastasis_losses):.4f} | Meta AUC: {metastasis_metrics['auc']:.4f} | Meta ACC: {metastasis_metrics['accuracy']:.4f} | "
          f"Status Loss: {np.mean(status_losses):.4f} | Status AUC: {status_metrics['auc']:.4f} | Status ACC: {status_metrics['accuracy']:.4f}")
    
    return  metastasis_metrics['auc'], status_metrics['auc']

def init_scheduler(args, optimizer):
    """Initialize learning rate scheduler."""
    if not args.use_scheduler:
        return None
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    return scheduler

if __name__ == "__main__":
    args = parser_args()

    # init setting
    warnings.filterwarnings("ignore")
    tensorboard_path, checkpoint_path, xlsx_path = init_output_directory(args.log_dir_path, args.log_name)
    seed_everything(args.seed)

    # init logger
    writer = SummaryWriter(log_dir=tensorboard_path, flush_secs=10)

    # init dataloader
    train_loader, val_loader, test_loader = init_dataloader(args)

    # init model
    model = Multitask_MILNET(backbone_name=args.backbone)
    model = model.cuda()

    # init optimizer and lr scheduler
    optimizer = init_optimizer(args, model)
    scheduler = init_scheduler(args, optimizer)

    # init loss functions
    metastasis_criterion = torch.nn.CrossEntropyLoss()
    status_criterion = torch.nn.CrossEntropyLoss()

    # training
    best_combined_auc = 0
    best_combined_epoch = 0
    best_meta_auc = 0
    best_meta_epoch = 0
    
    for epoch in range(1, args.epoch + 1):
     
        print(f"starts training epoch {epoch}")
        # Training
        train_meta_auc, train_status_auc = train_epoch(
            model, train_loader, optimizer, 
            metastasis_criterion, status_criterion, 
            args, writer, epoch
        )
        
        # Validation
        val_meta_auc, val_status_auc = evaluate(
            model, val_loader,
            metastasis_criterion, status_criterion,
            args, writer, epoch, phase='val'
        )
        
        if scheduler is not None:
            scheduler.step()
        
        # # Testing
        # test_meta_auc, test_status_auc = evaluate(
        #     model, test_loader,
        #     metastasis_criterion, status_criterion,
        #     args, writer, epoch, phase='test'
        # )

        # Combined metric for model selection (average of both AUCs)
        val_combined_auc = (val_meta_auc + val_status_auc) / 2

        # save best model based on combined validation AUC
        if val_combined_auc > best_combined_auc:
            best_combined_auc = val_combined_auc
            best_combined_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_meta_auc': val_meta_auc,
                'val_status_auc': val_status_auc,
                'val_combined_auc': val_combined_auc,
            }, os.path.join(checkpoint_path, "best_combined.pth"))
            print(f" Saved best combined model at epoch {epoch}")
        
        # save model based on best metasasis AUC
        if val_meta_auc > best_meta_auc:
            best_meta_auc = val_meta_auc
            best_meta_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_meta_auc': val_meta_auc,
                'val_status_auc': val_status_auc,
                'val_combined_auc': val_combined_auc,
            }, os.path.join(checkpoint_path, "best_meta.pth"))
             
        # save every 10 epochs
        if epoch % args.save_epoch_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f"epoch_{epoch}.pth"))
        
        if epoch == args.epoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, "last.pth"))

        print("-" * 120)
        torch.cuda.empty_cache()

    writer.close()
    print(f"Training completed!")
   
