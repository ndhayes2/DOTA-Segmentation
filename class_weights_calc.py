import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

def calculate_class_weights(mask_dir, num_classes):
    """
    Calculates class weights based on pixel frequency in the training masks.

    Args:
        mask_dir (str): Path to the directory containing training masks.
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    print(f"Calculating class weights from masks in: {mask_dir}")
    
    # Initialize a numpy array to store the pixel counts for each class
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))]
    
    # Loop through all mask files and count pixels
    for filename in tqdm(mask_files, desc="Counting Pixels"):
        mask_path = os.path.join(mask_dir, filename)
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not read mask {filename}")
            continue
            
        
        unique_classes, counts = np.unique(mask, return_counts=True)
        
        # Add the counts to the total running counts
        valid_indices = unique_classes < num_classes
        class_pixel_counts[unique_classes[valid_indices]] += counts[valid_indices]

    print("\n--- Pixel Counts ---")
    for i, count in enumerate(class_pixel_counts):
        print(f"Class {i}: {count} pixels")

    # --- Calculate Weights using Inverse Frequency ---
    total_pixels = np.sum(class_pixel_counts)
    
    class_weights = total_pixels / (num_classes * class_pixel_counts + 1e-6)
    
    
    class_weights /= np.sum(class_weights)
    
    print("\n--- Calculated Weights ---")
    print(class_weights)

    
    weights_tensor = torch.from_numpy(class_weights).float()
    return weights_tensor

# --- Main Execution ---
if __name__ == '__main__':
    
    TRAIN_MASK_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\train_split\masks"
    NUM_CLASSES = 16

    # Calculate the weights
    weights = calculate_class_weights(TRAIN_MASK_DIR, NUM_CLASSES)
    
    print("\n✅ Final PyTorch Weights Tensor:")
    print(weights)
    
    
    torch.save(weights, 'dota_class_weights.pt')