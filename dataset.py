import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class DotaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading DOTA image and mask patches.
    
    Args:
        image_dir (str): Path to the directory containing image patches.
        mask_dir (str): Path to the directory containing mask patches.
        transform (albumentations.Compose, optional): Augmentation pipeline.
    """
    

    

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
    
        
        all_mask_files = os.listdir(mask_dir)
        self.mask_files = [f for f in all_mask_files if f.endswith(".png")]

    

    def __len__(self):
        return len(self.mask_files)

        

  

    def __getitem__(self, idx):
        # Get the mask filename
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
    
       
        img_name = mask_name.replace('_mask.png', '.png')
        img_path = os.path.join(self.image_dir, img_name)
    
        # Load image and mask as NumPy arrays
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
    
        
        mask = mask.long()
            
        return image, mask