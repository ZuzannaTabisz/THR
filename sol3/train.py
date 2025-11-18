import gc
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# These imports assume that dataset.py, model.py, and utils.py are in the same directory.
from dataset import TrainDataset
from model import CustomModel, SoftCrossEntropyLoss
from utils import parse_images, merge_image_info, get_valid_transforms

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# This script implements the training strategy described by the Kaggle winner.
# Re-create the configuration from the inference script
class CFG:
    # General
    seed = 42
    num_workers = 0 # Set to 0 to prevent DataLoader hanging on the cluster
    
    # Data
    image_dir = './train_tiles' # This is the output of preprocess.py
    train_csv = '../data/train.csv'
    
    # Model
    model = 'swin_large_patch4_window12_384'
    model_reformat = False
    image_size = 384
    num_instance = 16 # Top 16 darkest tiles
    target_cols = ['CE', 'LAA']
    
    # Training
    epochs = 10
    n_fold = 5
    batch_size = 4
    lr = 1e-5
    
    # Output
    output_dir = f'./{model}-finetuned'

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_one_epoch(model, optimizer, scheduler, criterion, loader, device, scaler):
    model.train()
    losses = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        
        scaler.scale(loss).backward() # loss.backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
    return np.mean(losses)

@torch.no_grad()
def validate_one_epoch(model, criterion, loader, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Validating", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.append(loss.item())
        all_preds.append(y_preds.softmax(1).to('cpu').numpy())
        all_labels.append(labels.to('cpu').numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Konwersja etykiet one-hot na indeksy klas (0 dla CE, 1 dla LAA)
    y_true = np.argmax(all_labels, axis=1)
    # Konwersja prawdopodobieństw na przewidywane klasy
    y_pred_class = np.argmax(all_preds, axis=1)

    # Obliczanie metryk
    metrics = {}
    metrics['val_loss'] = np.mean(losses)
    metrics['log_loss'] = log_loss(y_true=y_true, y_pred=all_preds, labels=[0,1])
    metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['precision'] = precision_score(y_true, y_pred_class, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_class, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred_class, average='weighted', zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, all_preds[:, 1])
    except ValueError:
        # Może się zdarzyć w pierwszej epoce, jeśli model przewiduje tylko jedną klasę
        metrics['roc_auc'] = 0.0

    return metrics

def run_training(args):
    start_time = time.time()

    print("="*20)
    print("Training start")
    print("="*20)

    # Update CFG with the command-line argument for tile directory
    CFG.image_dir = args.tile_dir
    cfg = CFG
    set_seed(cfg.seed)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Data Preparation ---
    # This assumes a preprocessing script has already created tiles in `cfg.image_dir`
    # The author's description: "Tiling and pick the top 16 darkest tiles"
    train_info = pd.read_csv(cfg.train_csv)
    image_df = parse_images(cfg.image_dir)
    train_df = merge_image_info(image_df, train_info)

    # One-hot encode labels for the custom loss function
    train_df['CE'] = (train_df['label'] == 'CE').astype(float)
    train_df['LAA'] = (train_df['label'] == 'LAA').astype(float)

    # --- CV Strategy ---
    # Author's description: "5-fold Stratified Grouped KFold"
    df_for_split = train_df[train_df['instance_id'] == 0].reset_index(drop=True) # Use one row per patient for splitting
    skf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(skf.split(df_for_split, df_for_split['label'], df_for_split['patient_id'])):
        df_for_split.loc[val_idx, 'fold'] = fold
    
    train_df = train_df.merge(df_for_split[['image_id', 'fold']], on='image_id', how='left')

    # --- Training Loop ---
    for fold in range(cfg.n_fold):
        print(f"\n========== Starting Fold {fold+1}/{cfg.n_fold} ==========", flush=True)
        
        # Create datasets for the current fold
        train_fold_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
        valid_fold_df = train_df[train_df['fold'] == fold].reset_index(drop=True)
        
        train_dataset = TrainDataset(cfg, train_fold_df, get_valid_transforms(cfg))
        valid_dataset = TrainDataset(cfg, valid_fold_df, get_valid_transforms(cfg))
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        
        # --- Model, Loss, Optimizer ---
        # Author's description: "swin_large... + customized head", "replace average pooling with attention pooling"
        model = CustomModel(cfg, pretrained=True).to(device) # Start with ImageNet pre-trained weights

        # Author's description: "implement loss\metric following competition metric"
        criterion = SoftCrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler(device='cuda', enabled=True)
        
        best_score = float('inf')
        
        for epoch in range(cfg.epochs):
            print(f"--- Fold {fold+1}, Epoch {epoch+1}/{cfg.epochs} ---", flush=True)
            
            train_loss = train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device, scaler)
            val_metrics = validate_one_epoch(model, criterion, valid_loader, device)
            
            competition_score = val_metrics['log_loss']
            
            # Wyświetlanie wyników
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val LogLoss (Score): {val_metrics['log_loss']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1_score']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val ROC AUC: {val_metrics['roc_auc']:.4f}", flush=True)
            
            scheduler.step()
            
            if competition_score < best_score:
                best_score = competition_score
                print(f"    -> Nowy najlepszy wynik ({best_score:.4f})! Zapisywanie modelu dla foldu {fold+1}...", flush=True)
                torch.save({'model': model.state_dict()}, f"{cfg.output_dir}/{cfg.model}_fold{fold}_best.pth")
        
        del model, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal training time: {total_time / 3600:.2f} hours ({total_time / 60:.2f} minutes)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on preprocessed tiles.')
    parser.add_argument('--tile_dir', type=str, default='./train_tiles', help='Directory containing the preprocessed image tiles.')
    args = parser.parse_args()
    run_training(args)
