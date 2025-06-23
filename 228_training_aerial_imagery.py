# Modified script for aerial imagery training with new folder structure
"""
Modified version to handle folder structure:
archive-2/
├── train/
│   ├── *.jpg (aerial images)
│   └── *.png (masks)  
├── test/
│   └── *.jpg (aerial images only)
└── valid/
    └── *.jpg (aerial images only)

Color mapping for masks:
Urban land: 0,255,255 (Cyan)
Agriculture land: 255,255,0 (Yellow) 
Rangeland: 255,0,255 (Magenta)
Forest land: 0,255,0 (Green)
Water: 0,0,255 (Blue)
Barren land: 255,255,255 (White)
Unknown: 0,0,0 (Black)
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import glob
from collections import defaultdict

scaler = MinMaxScaler()

root_directory = 'archive-2/'
patch_size = 256

def get_image_dimensions(folder_path, extension="*.jpg"):
    """Find the highest and lowest dimensions of images in a folder"""
    dimensions = []
    files = glob.glob(os.path.join(folder_path, extension))
    
    for file_path in files:
        img = cv2.imread(file_path)
        if img is not None:
            height, width = img.shape[:2]
            dimensions.append((width, height, os.path.basename(file_path)))
    
    if not dimensions:
        return None, None, None, None
    
    # Sort by area (width * height)
    dimensions.sort(key=lambda x: x[0] * x[1])
    
    min_dim = dimensions[0]  # (width, height, filename)
    max_dim = dimensions[-1]
    
    return min_dim, max_dim, len(dimensions), dimensions

def analyze_dataset_dimensions():
    """Analyze dimensions across all subfolders"""
    print("=" * 60)
    print("DATASET DIMENSION ANALYSIS")
    print("=" * 60)
    
    all_stats = {}
    
    for subfolder in ['train', 'test', 'valid']:
        folder_path = os.path.join(root_directory, subfolder)
        if os.path.exists(folder_path):
            print(f"\n{subfolder.upper()} folder analysis:")
            print("-" * 30)
            
            # Analyze images
            min_dim, max_dim, count, all_dims = get_image_dimensions(folder_path, "*.jpg")
            if min_dim:
                print(f"Images (.jpg): {count} files")
                print(f"  Smallest: {min_dim[0]}x{min_dim[1]} ({min_dim[2]})")
                print(f"  Largest:  {max_dim[0]}x{max_dim[1]} ({max_dim[2]})")
                all_stats[f'{subfolder}_images'] = {'min': min_dim, 'max': max_dim, 'count': count}
            
            # Analyze masks (only for train folder)
            if subfolder == 'train':
                min_dim_mask, max_dim_mask, count_mask, all_dims_mask = get_image_dimensions(folder_path, "*.png")
                if min_dim_mask:
                    print(f"Masks (.png): {count_mask} files")
                    print(f"  Smallest: {min_dim_mask[0]}x{min_dim_mask[1]} ({min_dim_mask[2]})")
                    print(f"  Largest:  {max_dim_mask[0]}x{max_dim_mask[1]} ({max_dim_mask[2]})")
                    all_stats[f'{subfolder}_masks'] = {'min': min_dim_mask, 'max': max_dim_mask, 'count': count_mask}
        else:
            print(f"\n{subfolder.upper()} folder: NOT FOUND")
    
    return all_stats

# Analyze dataset dimensions
dimension_stats = analyze_dataset_dimensions()

# Define new color mapping (RGB format)
Urban_land = np.array([0, 255, 255])      # Cyan
Agriculture_land = np.array([255, 255, 0]) # Yellow  
Rangeland = np.array([255, 0, 255])       # Magenta
Forest_land = np.array([0, 255, 0])       # Green
Water = np.array([0, 0, 255])             # Blue
Barren_land = np.array([255, 255, 255])   # White
Unknown = np.array([0, 0, 0])             # Black

def rgb_to_2D_label(label):
    
    # Convert RGB mask to integer labels
    
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    
    # Use tolerance for color matching due to potential compression artifacts
    tolerance = 10
    
    def color_matches(pixel, target_color, tol=tolerance):
        return np.all(np.abs(pixel - target_color) <= tol)
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            pixel = label[i, j]
            
            if color_matches(pixel, Urban_land):
                label_seg[i, j] = 0
            elif color_matches(pixel, Agriculture_land):
                label_seg[i, j] = 1
            elif color_matches(pixel, Rangeland): 
                label_seg[i, j] = 2
            elif color_matches(pixel, Forest_land):
                label_seg[i, j] = 3
            elif color_matches(pixel, Water):
                label_seg[i, j] = 4
            elif color_matches(pixel, Barren_land):
                label_seg[i, j] = 5
            else:  # Unknown/Other
                label_seg[i, j] = 6
    
    return label_seg

def load_train_data():
    #Load training images and masks from train folder
    train_path = os.path.join(root_directory, 'train')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train folder not found: {train_path}")
    
    # Get all jpg files
    image_files = glob.glob(os.path.join(train_path, "*.jpg"))
    mask_files = glob.glob(os.path.join(train_path, "*.png"))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks in train folder")

    # Add these debug lines:
    print("\nFirst 5 image files:")
    for i, img in enumerate(image_files[:5]):
        print(f"  {i+1}. {os.path.basename(img)}")

    print("\nFirst 5 mask files:")
    for i, mask in enumerate(mask_files[:5]):
        print(f"  {i+1}. {os.path.basename(mask)}")
    
    # Match images with masks (assuming similar naming convention)
    image_dataset = []
    mask_dataset = []
    
    processed_pairs = 0
    
    
    for img_path in image_files:
        # Extract base number from image name (e.g., "2334" from "2334_sat.jpg")
        img_name = os.path.basename(img_path)  # "2334_sat.jpg"
        img_number = img_name.replace('_sat.jpg', '')  # "2334"
        
        # Look for corresponding mask (e.g., "2334_mask.png")
        expected_mask_name = f"{img_number}_mask.png"
        potential_masks = [m for m in mask_files if os.path.basename(m) == expected_mask_name]
        
        if potential_masks:
            mask_path = potential_masks[0]  # Take first match
            
            # Load and process image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load and process mask
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            # Ensure same dimensions
            if image.shape[:2] != mask.shape[:2]:
                print(f"Warning: Size mismatch for {img_name}. Image: {image.shape[:2]}, Mask: {mask.shape[:2]}")
                # Resize mask to match image
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Crop to nearest size divisible by patch_size
            SIZE_X = (image.shape[1] // patch_size) * patch_size
            SIZE_Y = (image.shape[0] // patch_size) * patch_size
            
            if SIZE_X > 0 and SIZE_Y > 0:
                image_cropped = image[:SIZE_Y, :SIZE_X]
                mask_cropped = mask[:SIZE_Y, :SIZE_X]
                
                # Extract patches
                print(f"Processing {img_name}: {image.shape} -> {SIZE_Y}x{SIZE_X}")
                
                patches_img = patchify(image_cropped, (patch_size, patch_size, 3), step=patch_size)
                patches_mask = patchify(mask_cropped, (patch_size, patch_size, 3), step=patch_size)
                
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        # Process image patch
                        single_patch_img = patches_img[i, j, 0]
                        single_patch_img = scaler.fit_transform(
                            single_patch_img.reshape(-1, single_patch_img.shape[-1])
                        ).reshape(single_patch_img.shape)
                        
                        # Process mask patch  
                        single_patch_mask = patches_mask[i, j, 0]
                        
                        image_dataset.append(single_patch_img)
                        mask_dataset.append(single_patch_mask)
                
                processed_pairs += 1
            else:
                print(f"Skipping {img_name}: too small after cropping")
        else:
            print(f"Warning: No mask found for {img_name}")
    
    print(f"Successfully processed {processed_pairs} image-mask pairs")
    print(f"Total patches created: {len(image_dataset)}")
    
    return np.array(image_dataset), np.array(mask_dataset)

def load_test_valid_data(subset='test'):
    # Load test or validation images (no masks available)
    subset_path = os.path.join(root_directory, subset)
    
    if not os.path.exists(subset_path):
        print(f"Warning: {subset} folder not found: {subset_path}")
        return np.array([])
    
    image_files = glob.glob(os.path.join(subset_path, "*.jpg"))
    print(f"Found {len(image_files)} images in {subset} folder")
    
    image_dataset = []
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop to nearest size divisible by patch_size
        SIZE_X = (image.shape[1] // patch_size) * patch_size
        SIZE_Y = (image.shape[0] // patch_size) * patch_size
        
        if SIZE_X > 0 and SIZE_Y > 0:
            image_cropped = image[:SIZE_Y, :SIZE_X]
            
            patches_img = patchify(image_cropped, (patch_size, patch_size, 3), step=patch_size)
            
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, 0]
                    single_patch_img = scaler.fit_transform(
                        single_patch_img.reshape(-1, single_patch_img.shape[-1])
                    ).reshape(single_patch_img.shape)
                    
                    image_dataset.append(single_patch_img)
    
    return np.array(image_dataset)

# Load training data
print("Loading training data...")
image_dataset, mask_dataset = load_train_data()

# Sanity check - view few images and masks
if len(image_dataset) > 0 and len(mask_dataset) > 0:
    import random
    image_number = random.randint(0, len(image_dataset) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Training Image')
    plt.imshow(image_dataset[image_number])
    plt.subplot(122)
    plt.title('Training Mask (RGB)')
    plt.imshow(mask_dataset[image_number])
    plt.show()

# Convert masks to integer labels
print("Converting masks to integer labels...")
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Unique labels in dataset:", np.unique(labels))

# Another sanity check with converted labels
if len(image_dataset) > 0:
    image_number = random.randint(0, len(image_dataset) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Training Image')
    plt.imshow(image_dataset[image_number])
    plt.subplot(122)
    plt.title('Training Label (Integer)')
    plt.imshow(labels[image_number][:,:,0], cmap='tab10')
    plt.colorbar()
    plt.show()

# Prepare for training
n_classes = len(np.unique(labels))
print(f"Number of classes: {n_classes}")

from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, labels_cat, test_size=0.20, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Load additional test/validation data if needed
test_images = load_test_valid_data('test')
valid_images = load_test_valid_data('valid')

print(f"Additional test images: {test_images.shape}")
print(f"Additional validation images: {valid_images.shape}")

print("\nDataset preparation complete!")
print("Next steps:")
print("1. Check the color mapping by examining some converted labels")
print("2. Consider class balancing - check distribution of classes")
print("3. Proceed with model training using the prepared data")
