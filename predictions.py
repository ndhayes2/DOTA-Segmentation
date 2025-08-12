import torch
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader


from model import UNet

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
VAL_IMG_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\val_split\images"
VAL_MASK_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\val_split\masks"
OUTPUT_DIR = "predictions"
NUM_CLASSES = 16

# Class names for the final report (Class 0 is background)
CLASS_NAMES = [
    'background', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
    'basketball-court', 'storage-tank', 'soccer-ball-field', 
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]



def evaluate_model(loader, model, device="cuda"):
    """Calculates evaluation metrics for the model on a given dataset."""
    num_correct = 0
    num_pixels = 0
    
    
    total_intersection = torch.zeros(NUM_CLASSES).to(device)
    total_union = torch.zeros(NUM_CLASSES).to(device)
    
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device) # y is the ground truth mask
            
            preds = torch.argmax(model(x), dim=1)
            
            # --- Calculate Pixel Accuracy ---
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # --- Calculate Intersection and Union for IoU ---
            for cls in range(NUM_CLASSES):
                pred_inds = (preds == cls)
                target_inds = (y == cls)
                intersection = (pred_inds & target_inds).sum()
                union = (pred_inds | target_inds).sum()
                
                total_intersection[cls] += intersection
                total_union[cls] += union

    # --- Final Metrics Calculation ---
    pixel_accuracy = (num_correct / num_pixels) * 100
    # Add a small epsilon to avoid division by zero and black holes opening
    class_iou = total_intersection / (total_union + 1e-6) 
    mean_iou = class_iou.mean()

    # --- Print Report ---
    print("\n--- Evaluation Report ---")
    print(f"Overall Pixel Accuracy: {pixel_accuracy:.2f}%")
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    print("\n--- IoU for each class ---")
    for i, iou in enumerate(class_iou):
        print(f"  {CLASS_NAMES[i]:<20}: {iou:.4f}")
    print("-------------------------\n")


def main():
    print(f"✅ Using device: {DEVICE}")

    # --- 1. Load Model ---
    model = UNet(in_channels=3, out_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # --- 2. Create DataLoader ---
    val_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    
    from dataset import DotaDataset 
    val_ds = DotaDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # --- 3. Run Evaluation ---
    evaluate_model(val_loader, model, device=DEVICE)

    


if __name__ == "__main__":
    main()