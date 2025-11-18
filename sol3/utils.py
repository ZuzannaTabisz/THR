import pyvips
import os
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def tile(img, sz=128, N=16):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=255)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < N:
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    return img


def save_dataset(df, N=16, max_size=20000, crop_size=1024, image_dir='./input/test', out_dir='./test'):
    format_to_dtype = {
        'uchar': np.uint8, 'char': np.int8, 'ushort': np.uint16, 'short': np.int16,
        'uint': np.uint32, 'int': np.int32, 'float': np.float32, 'double': np.float64,
        'complex': np.complex64, 'dpcomplex': np.complex128,
    }

    def vips2numpy(vi):
        return np.ndarray(
            buffer=vi.write_to_memory(),
            dtype=format_to_dtype[vi.format],
            shape=[vi.height, vi.width, vi.bands])

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    tk0 = tqdm(enumerate(df["image_id"].values), total=len(df))
    for i, image_id in tk0:
        image_path = f'{image_dir}/{image_id}.tif'
        if not os.path.exists(image_path):
            continue
        image = pyvips.Image.thumbnail(image_path, max_size)
        image = vips2numpy(image)
        images = tile(image, sz=crop_size, N=N)
        for idx, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{out_dir}/{image_id}_{idx}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        del img, image, images
        gc.collect()


def parse_images(folder):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    records = []

    for full_path in all_files:
        filename = os.path.basename(full_path)
        if '_' not in filename:
            continue  # skip malformed filenames

        parts = filename.replace('.jpg', '').split('_')
        if len(parts) < 3:
            continue  # not enough parts to extract image_id and instance_id

        try:
            # First two parts (joined) are usually the image_id
            image_id = '_'.join(parts[:-1])
            instance_id = int(parts[-1])
            records.append({
                'image_path': full_path,
                'image_id': image_id,
                'instance_id': instance_id
            })
        except ValueError:
            # Skip files where the last part is not a number (e.g., .txt disguised or wrong naming)
            print(f"Skipping invalid file (cannot parse instance_id): {filename}")
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(['image_id', 'instance_id']).reset_index(drop=True)
    return df


def merge_image_info(image_df, info_df):
    return image_df.merge(info_df, on='image_id', how='left').reset_index(drop=True)


def get_valid_transforms(cfg):
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=[0], std=[1], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count