import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from class_weights_calc import calculate_class_weights
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm # sexy progress bar


from dataset import DotaDataset
from model import UNet

# --- 1. Hyperparameters & Configuration ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 
NUM_EPOCHS = 50 # Number of times to loop over the dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")
NUM_CLASSES = 16 # 15 DOTA classes + 1 background class
scaler = GradScaler()



# --- 2. File Paths ---
TRAIN_IMG_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\train_split\images"
TRAIN_MASK_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\train_split\masks"
VAL_IMG_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\val_split\images"
VAL_MASK_DIR = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\val_split\masks"


def check_accuracy(loader, model, device="cuda"):
    """
    Helper function to check accuracy on the validation set.
    Calculates pixel accuracy and Dice score (a common metric for segmentation).
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.argmax(model(x), dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds == y).sum()) / (
                (preds == preds).sum() + (y == y).sum()
            )
    accuracy = num_correct/num_pixels
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train() 
    return accuracy


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """
    A single epoch of training using Automatic Mixed Precision (AMP).
    """
    loop = tqdm(loader, desc="Training")
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.long().to(device=device)

        
        with autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        
        scaler.step(optimizer)
        
        
        scaler.update()

        
        loop.set_postfix(loss=loss.item())


def main():
    """
    Main function to tie everything together.
    """
    # --- Data Augmentation ---
    train_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # --- Datasets and DataLoaders ---
    train_ds = DotaDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    scaler = GradScaler()

    val_ds = DotaDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Model, Loss, and Optimizer ---
    model = UNet(in_channels=3, out_classes=NUM_CLASSES).to(DEVICE)
    print("Loading pre-calculated weights...")
    weights = torch.load('dota_class_weights.pt').to(DEVICE) 
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Variable to track best performance ---
    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

        # Check accuracy on validation set
        current_accuracy = check_accuracy(val_loader, model, device=DEVICE)

        # Save model if it's the best one 
        if current_accuracy > best_val_accuracy:
            best_val_accuracy = current_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ New best model saved!")


if __name__ == "__main__":
    main()