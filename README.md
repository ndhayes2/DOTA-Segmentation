# Semantic Segmentation of DOTAv1.0 dataset with Python

## Table of Contents
- [Project Overview](#project-overview)
-  [Key Features](#key-features)
- [Problem Statement](#problem-statement)
- [Data Pipeline](#data-pipeline-handling-gigapixel-images)
- [Model & Training](#model-and-training)
- [Results & Analysis](#results--analysis)
- [How to Use](#how-to-use)
- [Tech Stack](#tech-stack)

## Project Overview
This project implements a U-Net model from scratch in PyTorch to perform semantic segmentation on the DOTA v1.0 dataset. The primary goal was not just to build a model, but to engineer solutions for two common real-world computer vision problems: processing images that are too large to fit into memory and training a model on a dataset with a highly imbalanced class distribution.

## Key Features 
- **U-Net from Scratch**: Implementation of the U-Net architecture in PyTorch.
-  **Large Image Processing**: A custom data pipeline to patch gigapixel-sized satellite images into trainable segments.
-  **Class Imbalance Solution**: Use of a custom weighted cross-entropy loss function to improve learning on rare classes.
-  **Data Augmentation**: Targeted augmentation strategies to increase model generalization.
-  **End-to-End Pipeline**: Scripts for data preparation, training, evaluation, and visual prediction.

## Problem Statement
Training a segmentation model on the DOTA dataset presents two main challenges:
1. **Image Size**: The original images are massive (up to 4000x4000 pixels) and cannot be loaded directly into GPU memory.
 2. **Class Imbalance**: The dataset is severely imbalanced. For instance, 'small-vehicle' and 'large-vehicle' pixels vastly outnumber pixels for classes like 'bridge' or 'harbor'.

<img width="4200" height="2400" alt="class_distribution" src="https://github.com/user-attachments/assets/f7304818-04d9-47c0-ac99-8cbd3ddb7674" />


## Data Pipeline: Handling Gigapixel Images
To solve the image size problem, I engineered a data pipeline that crops the original large images and their corresponding masks into smaller, overlapping 1024x1024 patches. This approach allows the model to train efficiently on segments of the image without losing spatial context at the patch borders. The pipeline generated a dataset of over 10,000 trainable patches.

## Model and Training
### Architecture
The project uses the U-Net architecture, which is renowned for its performance in biomedical and satellite image segmentation. Its encoder-decoder structure with skip connections allows it to capture both high-level context and fine-grained detail.
### Tackling Class Imbalance
The key to improving performance was addressing the class imbalance. I implemented a custom **weighted cross-entropy loss function**. Weights for each class were calculated based on the inverse of their pixel frequency, forcing the model to pay more attention to underrepresented classes.

$L_{wce} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} w_c \cdot y_{i,c} \log(p_{i,c})$

This, combined with targeted data augmentation (rotations, flips, color jitter), was crucial for the results.

## Results & Analysis
The model's performance was evaluated using Mean Intersection over Union (mIoU). The strategies implemented resulted in a **180% relative increase in mIoU** compared to a baseline model.

<img width="412" height="62" alt="image" src="https://github.com/user-attachments/assets/0b4b547b-a533-4f70-ac69-8d5c5713d4ec" />


### Future Work
- Experiment with more advanced architectures like DeepLabv3+.
-  Employ more sophisticated data augmentation techniques (e.g., Copy-Paste).
-  Train for more epochs on a more powerful GPU.

## Tech Stack
- Python 3.8+
-   PyTorch
-  OpenCV
 - NumPy 
 -  Matplotlib
-   Scikit-learn
