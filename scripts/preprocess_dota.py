import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import subprocess
import shutil
print('imports done')


#Configuration
DOTA_BASE_PATH = r"C:\Users\Nikolas\Desktop\dota_segmentation\DOTA v1.0"
DEV_KIT_PATH = r"C:\Users\Nikolas\Desktop\dota_segmentation\DOTA_devkit-master"
sys.path.append(DEV_KIT_PATH)
import ImgSplit_multi_process as dota_splitter
OUTPUT_SPLIT_PATH = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed"

#Patching Params
IMG_SIZE = 1024   #size of square 'patches', 1024x1024 pixels
OVERLAP_GAP = 200 #overlap between patches, 200 pixels


CLASS_NAMES_V1_0 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES_V1_0)}
ID_TO_CLASS = {i + 1: name for i, name in enumerate(CLASS_NAMES_V1_0)}

print("DOTA Class to ID Mapping:")
for cls_name, cls_id in CLASS_TO_ID.items():
    print(f"  {cls_name}: {cls_id}")
print("-" * 30)


# --- Helper Function to Convert OBB to Mask ---
def create_mask_from_obb(image_shape, obb_coords, class_id):
    """
    Creates a binary mask for a single object from its oriented bounding box coordinates.
    Args:
        image_shape (tuple): (height, width) of the image.
        obb_coords (list): 8 float coordinates [x1, y1, x2, y2, x3, y3, x4, y4].
        class_id (int): The integer ID of the class to fill the mask with.
    Returns:
        numpy.ndarray: A grayscale mask (H, W) with pixel values of class_id within the polygon, 0 otherwise.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    
    points = np.array(obb_coords).reshape(4, 2).astype(np.int32)
    
    
    cv2.fillPoly(mask, [points], class_id)
    return mask


# --- Main Preprocessing Function ---
def preprocess_dota_dataset(data_type, dev_kit_base_path, dota_base_path, output_base_path, img_size, overlap_gap, class_to_id_map):
    """
    Orchestrates the DOTA dataset preprocessing.
    """
    print(f"\n--- Starting preprocessing for {data_type} set ---")

    src_path = os.path.join(dota_base_path, data_type)
    output_path = os.path.join(output_base_path, f'{data_type}_split')
    output_split_images_path = os.path.join(output_path, 'images')
    output_split_masks_path = os.path.join(output_path, 'masks')
    
    os.makedirs(output_split_images_path, exist_ok=True)
    os.makedirs(output_split_masks_path, exist_ok=True)

    # --- Step 1: Patching Images using DOTA_devkit ---
    print(f"Step 1/2: Splitting {data_type} images...")
    try:
        splitter = dota_splitter.splitbase(
            basepath=src_path,
            outpath=output_path,
            gap=overlap_gap,
            subsize=img_size,
            ext='.png'
        )
        splitter.splitdata(rate=1)
    except Exception as e:
        print(f"An error occurred during image splitting: {e}")
        return False
    print("Image splitting completed.")

    # --- Step 2: Convert OBB annotations to Masks ---
    print("Step 2/2: Converting annotations to masks...")
    split_labels_path = os.path.join(output_path, 'labelTxt')

    if not os.path.isdir(split_labels_path):
        print(f"Warning: Label directory '{split_labels_path}' not found. No masks will be generated for {data_type} set.")
        return True 

    split_label_files = os.listdir(split_labels_path)
    
    for label_filename in split_label_files:
        if not label_filename.endswith('.txt'):
            continue
        
        base_filename = os.path.splitext(label_filename)[0]
        image_path = os.path.join(output_split_images_path, f"{base_filename}.png")
        label_path = os.path.join(split_labels_path, label_filename)
        
        if not os.path.exists(image_path):
            continue
            
        try:
            img = cv2.imread(image_path)
            if img is None: continue
            
            img_height, img_width = img.shape[:2]
            current_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            object_count = 0

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9: continue
                    
                    object_count += 1
                    class_name = parts[8].lower()
                    class_id = class_to_id_map.get(class_name, 0)
                    obb_coords = [float(p) for p in parts[:8]]
                    
                    object_mask = create_mask_from_obb((img_height, img_width), obb_coords, class_id)
                    current_mask = np.maximum(current_mask, object_mask)

            if object_count > 0:
                mask_output_path = os.path.join(output_split_masks_path, f"{base_filename}_mask.png")
                cv2.imwrite(mask_output_path, current_mask)

        except Exception as e:
            print(f"Error processing {label_filename}: {e}")
            continue

    print(f"Mask generation for {data_type} set completed.")
    
    # Clean up temporary split labels folder
    print(f"Cleaning up temporary labelTxt folder: {split_labels_path}")
    shutil.rmtree(split_labels_path)
    
    
    return True


# --- Main Execution ---
if __name__ == "__main__":
    # Process training set
    success_train = preprocess_dota_dataset(
        data_type='train',
        dev_kit_base_path=DEV_KIT_PATH,
        dota_base_path=DOTA_BASE_PATH,
        output_base_path=OUTPUT_SPLIT_PATH,
        img_size=IMG_SIZE,
        overlap_gap=OVERLAP_GAP,
        class_to_id_map=CLASS_TO_ID
    )

    if success_train:
        # Process validation set
        success_val = preprocess_dota_dataset(
            data_type='val',
            dev_kit_base_path=DEV_KIT_PATH,
            dota_base_path=DOTA_BASE_PATH,
            output_base_path=OUTPUT_SPLIT_PATH,
            img_size=IMG_SIZE,
            overlap_gap=OVERLAP_GAP,
            class_to_id_map=CLASS_TO_ID
        )
        if success_val:
            print("\nAll DOTA dataset preprocessing completed successfully!")
        else:
            print("\nValidation set preprocessing failed. Check logs above.")
    else:
        print("\nTraining set preprocessing failed. Check logs above.")