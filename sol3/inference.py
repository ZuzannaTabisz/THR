import os
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
import cv2
import pyvips
 
from sklearn.metrics import (log_loss, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, classification_report)

# Importy z istniejących plików projektu
from dataset import TrainDataset
from model import CustomModel
from utils import parse_images, merge_image_info, get_valid_transforms
from train import CFG # Importujemy konfigurację z pliku treningowego

def run_preprocessing(cfg, raw_image_dir, csv_path, tile_dir):
    """
    Przetwarza surowe obrazy .tif na kafelki, tak jak w preprocess.py.
    """
    print(f"\n--- Uruchamianie preprocessingu ---")
    print(f"Katalog wejściowy: {raw_image_dir}")
    print(f"Katalog wyjściowy na kafelki: {tile_dir}")

    os.makedirs(tile_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].unique()

    pbar = tqdm(image_ids, desc="Preprocessing obrazów")
    for image_id in pbar:
        image_path = os.path.join(raw_image_dir, f"{image_id}.tif")
        if not os.path.exists(image_path):
            print(f"Ostrzeżenie: Pomijam {image_id}, plik nie istnieje w {raw_image_dir}")
            continue

        try:
            image = pyvips.Image.new_from_file(image_path, access="sequential")
            
            # Dzielenie na kafelki i obliczanie "ciemności"
            tiles = []
            for y in range(0, image.height, cfg.image_size):
                for x in range(0, image.width, cfg.image_size):
                    if x + cfg.image_size > image.width or y + cfg.image_size > image.height:
                        continue
                    tile = image.crop(x, y, cfg.image_size, cfg.image_size)
                    np_tile = np.ndarray(buffer=tile.write_to_memory(), dtype=np.uint8, shape=[tile.height, tile.width, tile.bands])
                    
                    if np_tile.shape[2] == 1: # Obrazy w skali szarości
                        darkness = np_tile.mean()
                    else: # Obrazy kolorowe
                        darkness = cv2.cvtColor(np_tile, cv2.COLOR_RGB2GRAY).mean()
                    tiles.append((darkness, np_tile))
            
            # Sortowanie i wybór 16 najciemniejszych
            tiles.sort(key=lambda t: t[0])
            for i, (_, tile_data) in enumerate(tiles[:cfg.num_instance]):
                output_path = os.path.join(tile_dir, f"{image_id}_{i:04d}.png")
                cv2.imwrite(output_path, cv2.cvtColor(tile_data, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Błąd podczas przetwarzania {image_id}: {e}")

@torch.no_grad()
def run_inference(cfg, test_df, device):
    """
    Uruchamia inferencję dla wszystkich 5 foldów i uśrednia wyniki.
    """
    test_dataset = TrainDataset(cfg, test_df, get_valid_transforms(cfg))
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    all_fold_preds = []

    for fold in range(cfg.n_fold):
        print(f"\n--- Inferencja dla Fold {fold+1}/{cfg.n_fold} ---", flush=True)
        
        model = CustomModel(cfg, pretrained=False).to(device)
        model_path = f"{cfg.output_dir}/{cfg.model}_fold{fold}_best.pth"
        
        try:
            state_dict = torch.load(model_path, map_location=device)['model']
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku modelu dla foldu {fold}: {model_path}")
            print("Upewnij się, że trening został zakończony i modele są w odpowiednim katalogu.")
            return None, None

        model.eval()
        fold_preds = []
        pbar = tqdm(test_loader, desc=f"Predykcja Fold {fold+1}", leave=False)
        for images, _ in pbar:
            images = images.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=True):
                y_preds = model(images)
            
            fold_preds.append(y_preds.softmax(1).to('cpu').numpy())
        
        all_fold_preds.append(np.concatenate(fold_preds))

    # Uśrednienie predykcji ze wszystkich foldów
    avg_preds = np.mean(all_fold_preds, axis=0)
    return avg_preds

def print_metrics(y_true, y_pred_probs):
    """
    Oblicza i wyświetla kompleksowy zestaw metryk ewaluacyjnych.
    """
    # Konwersja prawdopodobieństw na klasy (0 lub 1)
    y_pred_class = np.argmax(y_pred_probs, axis=1)
    
    print("\n" + "="*30)
    print("      WYNIKI EWALUACJI MODELU")
    print("="*30 + "\n")

    # 1. Metryki podstawowe
    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, average='weighted')
    recall = recall_score(y_true, y_pred_class, average='weighted')
    f1 = f1_score(y_true, y_pred_class, average='weighted')
    
    print("--- Podstawowe metryki ---")
    print(f"Dokładność (Accuracy): {accuracy:.4f}")
    print(f"Precyzja (Precision):  {precision:.4f}")
    print(f"Czułość (Recall):      {recall:.4f}")
    print(f"F1-Score:              {f1:.4f}\n")

    # 2. Metryki probabilistyczne
    try:
        # Metryka z konkursu Kaggle
        ll = log_loss(y_true, y_pred_probs, labels=[0, 1])
        # AUC
        roc_auc = roc_auc_score(y_true, y_pred_probs[:, 1])
        print("--- Metryki probabilistyczne ---")
        print(f"Log Loss (metryka konkursowa): {ll:.4f}")
        print(f"ROC AUC Score:                   {roc_auc:.4f}\n")
    except ValueError as e:
        print(f"Nie można obliczyć Log Loss lub ROC AUC: {e}")

    # 3. Macierz pomyłek (Confusion Matrix)
    print("--- Macierz pomyłek ---")
    cm = confusion_matrix(y_true, y_pred_class)
    print("         Pred: CE | Pred: LAA")
    print(f"True: CE | {cm[0, 0]:<7d} | {cm[0, 1]:<7d}")
    print(f"True: LAA| {cm[1, 0]:<7d} | {cm[1, 1]:<7d}\n")

    # 4. Raport klasyfikacji
    print("--- Raport klasyfikacji ---")
    print(classification_report(y_true, y_pred_class, target_names=CFG.target_cols))

def main(args):
    # --- Krok 1: Preprocessing ---
    if not args.skip_preprocessing:
        run_preprocessing(CFG, args.raw_image_dir, args.csv_path, args.tile_dir)
    else:
        print("Pomijam krok preprocessingu, używam istniejących kafelków.")

    # --- Krok 2: Przygotowanie danych do inferencji ---
    cfg = CFG
    cfg.image_dir = args.tile_dir # Używamy kafelków testowych
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_info = pd.read_csv(args.csv_path)
    image_df = parse_images(cfg.image_dir)
    test_df = merge_image_info(image_df, test_info)

    # One-hot encode dla spójności z datasetem (tworzymy puste kolumny, jeśli nie istnieją)
    if 'label' in test_df.columns:
        test_df['CE'] = (test_df['label'] == 'CE').astype(float)
        test_df['LAA'] = (test_df['label'] == 'LAA').astype(float)
    else:
        test_df['CE'] = 0.0
        test_df['LAA'] = 0.0

    # --- Krok 3: Uruchomienie inferencji ---
    final_preds = run_inference(cfg, test_df, device)

    if final_preds is not None:
        # Przygotowanie ramek danych do zapisu wyników
        true_labels_df = test_df[test_df['instance_id'] == 0].reset_index(drop=True)
        
        # --- Krok 4: Ewaluacja lub zapis predykcji ---
        if 'label' in true_labels_df.columns:
            print("\nZnaleziono etykiety w pliku CSV. Obliczanie metryk wydajności.")
            y_true = (true_labels_df['label'] == 'LAA').astype(int).values # 0 dla CE, 1 dla LAA
            print_metrics(y_true, final_preds)
        else:
            print("\nBrak etykiet w pliku CSV. Zapisywanie predykcji do pliku.")
            pred_labels = [cfg.target_cols[i] for i in np.argmax(final_preds, axis=1)]
            submission_df = true_labels_df[['image_id']].copy()
            submission_df['label'] = pred_labels
            submission_df.to_csv(args.output_csv, index=False)
            print(f"Predykcje zostały zapisane w pliku: {args.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uruchamia preprocessing i inferencję na surowych danych.')
    parser.add_argument('--raw_image_dir', type=str, default='./data/test', help='Katalog z surowymi obrazami .tif do przetworzenia.')
    parser.add_argument('--csv_path', type=str, default='./data/test.csv', help='Ścieżka do pliku .csv z listą obrazów (np. test.csv).')
    parser.add_argument('--tile_dir', type=str, default='./test_tiles_generated', help='Katalog, w którym zostaną zapisane wygenerowane kafelki.')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Nazwa pliku CSV do zapisu wyników predykcji.')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Pomiń krok preprocessingu, jeśli kafelki już istnieją.')
    args = parser.parse_args()
    main(args)