import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def generate_class_imbalance_chart(masks_dir, num_classes, class_names, output_path="class_distribution.png"):
    """
    Analyzes segmentation masks to count pixel distribution per class and saves a chart.

    Args:
        masks_dir (str): Directory containing the ground truth mask images.
        num_classes (int): The total number of classes (e.g., 16 for DOTA).
        class_names (list): A list of strings with the name for each class ID.
        output_path (str): Path to save the output chart image.
    """
    if len(class_names) != num_classes:
        raise ValueError("The length of class_names must be equal to num_classes.")

    print(f"Analyzing masks in: {masks_dir}")

    # Initialize a numpy array to store pixel counts for each class
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.png', '.tif', '.jpg'))]

    # Use tqdm for a progress bar
    for filename in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(masks_dir, filename)
        try:
            with Image.open(mask_path) as mask_image:
                mask_array = np.array(mask_image)

                # Get unique class IDs and their counts in the current mask
                unique_classes, counts = np.unique(mask_array, return_counts=True)

                # Add the counts to our total
                for class_id, count in zip(unique_classes, counts):
                    if class_id < num_classes:
                        pixel_counts[class_id] += count
        except Exception as e:
            print(f"Could not process file {filename}: {e}")

    print("Analysis complete. Generating chart...")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create the bar chart
    bars = ax.bar(class_names, pixel_counts, color=plt.cm.get_cmap('tab20', num_classes).colors)
    
    # Use a logarithmic scale for the y-axis to better visualize large differences
    ax.set_yscale('log')

    # Add labels and title
    ax.set_ylabel('Total Pixel Count (Log Scale)', fontsize=12)
    ax.set_xlabel('Object Class', fontsize=12)
    ax.set_title('Class Distribution Across the Dataset', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval):,}', va='bottom', ha='center') # va: vertical alignment

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Chart saved to {output_path}")


if __name__ == '__main__':
    # --- Configuration: YOU NEED TO EDIT THIS ---
    
    # 1. Path to your directory containing the ground truth segmentation masks
    # These are the black-and-white images where pixel values are class IDs
    MASKS_DIRECTORY = r"C:\Users\Nikolas\Desktop\dota_segmentation\data_processed\train_split\masks" 

    # 2. Define your class names in the correct order (0 to 15)
    DOTA_CLASS_NAMES = [
        'background', 'plane', 'baseball-diamond', 'bridge',
        'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
        'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]
    
    # --- Run the function ---
    generate_class_imbalance_chart(
        masks_dir=MASKS_DIRECTORY,
        num_classes=len(DOTA_CLASS_NAMES),
        class_names=DOTA_CLASS_NAMES,
        output_path="class_distribution.png" # The name of the output file
    )