import torch
from torch.utils.data import Dataset
import cv2
import logging

# Configure simple logger (you can also use print if you prefer)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, cfg, csv, transform):
        self.cfg = cfg
        self.csv = csv.reset_index(drop=True)
        self.transform = transform
        self.image_ids = self.csv.image_id.unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        df = self.csv.loc[self.csv.image_id == img_id]

        images = []
        paths = df.image_path.values

        for i, path in enumerate(paths):
            if i >= self.cfg.num_instance:
                break
            try:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError("cv2.imread returned None")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = self.transform(image=image)
                images.append(transformed['image'])
            except Exception as e:
                log.warning(f"Error reading image {path} (image_id={img_id}, idx={i}): {e}")
                # Create white placeholder image with correct size
                placeholder = torch.zeros(3, self.cfg.image_size, self.cfg.image_size)
                images.append(placeholder)

        current_n = len(images)

        if current_n != self.cfg.num_instance:
            log.warning(
                f"image_id={img_id} has {current_n} tiles instead of {self.cfg.num_instance}. "
                f"Padding/duplicating to fix."
            )

            if current_n < self.cfg.num_instance:
                # Pad with the last available tile (or white if none)
                if current_n == 0:
                    filler = torch.zeros(3, self.cfg.image_size, self.cfg.image_size)
                else:
                    filler = images[-1]
                while len(images) < self.cfg.num_instance:
                    images.append(filler)

            else:  # current_n > num_instance (should not happen after the loop, but just in case)
                images = images[:self.cfg.num_instance]

        # Stack to (num_instance, C, H, W)
        images = torch.stack(images, dim=0)

        # Labels are the same for all tiles of one patient
        labels = df.iloc[0][self.cfg.target_cols].values.astype('float32')
        labels = torch.from_numpy(labels)

        return images, labels