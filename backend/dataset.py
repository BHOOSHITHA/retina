import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class RetinalDataset(Dataset):
    def __init__(self, root_dir, dataset_name="DRIVE", split="train", img_size=(256, 256)):
        self.root_dir = root_dir
        self.dataset_name = dataset_name.upper()
        self.split = split
        self.img_size = img_size
        self.image_paths = []
        self.mask_paths = []

        self._load_paths()

    def _load_paths(self):
        if self.dataset_name == "DRIVE":
            folder = "training" if self.split == "train" else "test"
            img_dir = os.path.join(self.root_dir, "DRIVE", folder, "images")
            mask_dir = os.path.join(self.root_dir, "DRIVE", folder, "1st_manual")
            
            self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.*")))

        elif self.dataset_name == "CHASE_DB1":
            img_dir = os.path.join(self.root_dir, "CHASE_DB1", "images")
            mask_dir = os.path.join(self.root_dir, "CHASE_DB1", "labels")
            
            all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            all_masks = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
            
            split_idx = int(len(all_imgs) * 0.8)
            if self.split == "train":
                self.image_paths = all_imgs[:split_idx]
                self.mask_paths = all_masks[:split_idx]
            else:
                self.image_paths = all_imgs[split_idx:]
                self.mask_paths = all_masks[split_idx:]

        elif self.dataset_name == "FIVES":
            folder = "train" if self.split == "train" else "test"
            img_dir = os.path.join(self.root_dir, "FIVES", folder, "Original")
            mask_dir = os.path.join(self.root_dir, "FIVES", folder, "Ground truth")
            
            self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not properly supported with ground truth.")

        if len(self.image_paths) == 0:
            print(f"Warning: No images found for {self.dataset_name} in {self.split}")
        elif len(self.image_paths) != len(self.mask_paths):
            print(f"Warning: Mismatch in images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) for {self.dataset_name}. Truncating to match.")
            min_len = min(len(self.image_paths), len(self.mask_paths))
            self.image_paths = self.image_paths[:min_len]
            self.mask_paths = self.mask_paths[:min_len]

    def preprocess_image(self, img_path):
        # Read image
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"Could not read {img_path}")
        
        # 1. Extract Green Channel
        b, g, r = cv2.split(img)
        
        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(g)
        
        # Merge back to fake 3-channel (so U-Net sees 3 channels)
        img_out = cv2.merge([g_clahe, g_clahe, g_clahe])
        
        # Resize
        img_out = cv2.resize(img_out, self.img_size)
        
        # Normalize to [0, 1]
        img_out = img_out.astype(np.float32) / 255.0
        
        # HWC to CHW for PyTorch
        img_out = np.transpose(img_out, (2, 0, 1))
        return torch.tensor(img_out, dtype=torch.float32)

    def preprocess_mask(self, mask_path):
        # Using PIL since some annotations are .gif which OpenCV struggles with
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Resize
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Binarize
        mask = (mask > 127).astype(np.float32)
        
        # Add channel dimension
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img_tensor = self.preprocess_image(img_path)
        mask_tensor = self.preprocess_mask(mask_path)
        
        return img_tensor, mask_tensor
