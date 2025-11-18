# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import pyvips
import gc
import numpy as np
from tqdm.auto import tqdm
import cv2
import time
from datetime import datetime


def log_message(log_file, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp}: {message}\n")
    print(message)


def run_preprocessing(args):
    print("=" * 60)
    print("Starting Preprocessing Stage")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(
        args.output_dir,
        f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    # Load CSV
    try:
        df = pd.read_csv(args.csv_path)
        log_message(log_file, f"Loaded {args.csv_path} - {len(df)} records")
    except Exception as e:
        log_message(log_file, f"Failed to load CSV: {e}")
        return

    # Find all .tif images
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith('.tif')]
    total = len(image_files)
    log_message(log_file, f"Found {total} .tif images in {args.image_dir}")

    tile_size = 224
    dark_threshold = 200
    min_dark_ratio = 0.6
    min_rbc_ratio = 0.05

    for idx, filename in enumerate(tqdm(image_files, total=total), start=1):
        image_id = os.path.splitext(filename)[0]
        image_path = os.path.join(args.image_dir, filename)

        # Skip if already processed
        if any(f.startswith(image_id + "_") or f.startswith(image_id + ".") for f in os.listdir(args.output_dir)):
            log_message(log_file, f"[{idx}/{total}] Skipping {filename} - already processed")
            continue

        # Get label (just for logging, not used for folder split)
        label = df[df['image_id'] == image_id]['label'].values
        label_str = label[0] if len(label) > 0 else "unknown"

        start_time = time.time()

        try:
            # Load with sequential access – memory efficient
            vips_img = pyvips.Image.new_from_file(image_path, access="sequential")
            log_message(log_file, f"[{idx}/{total}] Processing {filename} | {vips_img.width}x{vips_img.height} | Label: {label_str}")

            # Use only upper half (common in amyloidosis datasets)
            vips_img = vips_img.crop(0, 0, vips_img.width, vips_img.height // 2)

            # Pad to be divisible by tile_size
            pad_x = (tile_size - vips_img.width % tile_size) % tile_size
            pad_y = (tile_size - vips_img.height % tile_size) % tile_size
            vips_img = vips_img.embed(
                pad_x // 2, pad_y // 2,
                vips_img.width + pad_x,
                vips_img.height + pad_y,
                background=[255, 255, 255]
            )

            saved_count = 0
            for y in range(0, vips_img.height, tile_size):
                for x in range(0, vips_img.width, tile_size):
                    tile = vips_img.crop(x, y, tile_size, tile_size)

                    # Convert to numpy
                    np_tile = np.ndarray(
                        buffer=tile.write_to_memory(),
                        dtype=np.uint8,
                        shape=(tile.height, tile.width, tile.bands)
                    )

                    if np_tile.shape[2] < 3:
                        continue

                    # Dark tissue ratio
                    gray = cv2.cvtColor(np_tile[..., :3], cv2.COLOR_RGB2GRAY)
                    dark_ratio = np.mean(gray < dark_threshold)

                    # RBC ratio in HSV
                    hsv = cv2.cvtColor(np_tile[..., :3], cv2.COLOR_RGB2HSV)
                    rbc_mask = cv2.inRange(hsv, (0, 50, 50), (15, 255, 255))
                    rbc_ratio = np.mean(rbc_mask > 0)

                    # Keep only good tiles
                    if dark_ratio >= min_dark_ratio and rbc_ratio >= min_rbc_ratio:
                        bgr_tile = cv2.cvtColor(np_tile[..., :3], cv2.COLOR_RGB2BGR)
                        output_path = os.path.join(args.output_dir, f"{image_id}_{saved_count}.jpg")
                        cv2.imwrite(output_path, bgr_tile, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        saved_count += 1

            elapsed = time.time() - start_time
            log_message(log_file, f"[{idx}/{total}] Done {filename} - {saved_count} tiles saved ({elapsed:.1f}s)")

            # Cleanup
            del vips_img, tile, np_tile
            gc.collect()

        except Exception as e:
            log_message(log_file, f"[{idx}/{total}] ERROR {filename}: {e}")

    log_message(log_file, "Preprocessing completed successfully!")
    print("\nPreprocessing finished!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract high-quality 224x224 tiles from whole-slide .tif images.')
    parser.add_argument('--image_dir', type=str, default='./data/train',
                        help='Directory containing the raw .tif images')
    parser.add_argument('--csv_path', type=str, default='./data/train.csv',
                        help='Path to train.csv (with image_id and label columns)')
    parser.add_argument('--output_dir', type=str, default='./train_tiles',
                        help='Directory to save all extracted tiles (single folder)')
    parser.add_argument('--num_tiles', type=int, default=16,
                        help='Not used in this version (keeps all good tiles). Keep for compatibility.')

    args = parser.parse_args()
    run_preprocessing(args)