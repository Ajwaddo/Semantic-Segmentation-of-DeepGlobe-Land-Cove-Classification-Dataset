# ori_228_training_aerial_imagery.py (modified version

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from tqdm import tqdm
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from simple_multi_unet_model import multi_unet_model, jacard_coef
from keras.utils import to_categorical
import tensorflow as tf

scaler = MinMaxScaler()
patch_size = 256

# NEW: Set root directory to archive-2/train
root_directory = 'archive-2/train'

# NEW: Read satellite images (.jpg) and masks (.png)
image_dataset = []
mask_dataset = []

def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

image_files = sorted([f for f in os.listdir(root_directory) if f.endswith('_sat.jpg')])

for image_file in tqdm(image_files, desc="Processing images and masks"):
    image_path = os.path.join(root_directory, image_file)
    mask_path = image_path.replace('_sat.jpg', '_mask.png')

    image = cv2.imread(image_path)
    image = resize_image(image)  # Resize to 256x256
    image = image.astype(np.float32) / 255.0  # Normalize
    image_dataset.append(image)


    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = resize_image(mask)  # Resize to 256x256
    mask_dataset.append(mask)


image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# NEW: RGB color map replacement for new classes
Urban_land = [0, 255, 255]
Agriculture_land = [255, 255, 0]
Rangeland = [255, 0, 255]
Forest_land = [0, 255, 0]
Water = [0, 0, 255]
Barren_land = [255, 255, 255]
Unknown = [0, 0, 0]

COLOR_MAP = {
    0: Urban_land,
    1: Agriculture_land,
    2: Rangeland,
    3: Forest_land,
    4: Water,
    5: Barren_land,
    6: Unknown
}

COLOR_ARRAY = np.array(list(COLOR_MAP.values()))

# Convert RGB mask to class label

def rgb_to_2D_label(label):
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    for idx, color in enumerate(COLOR_ARRAY):
        label_seg[np.all(label == color, axis=-1)] = idx
    return label_seg[:, :, 0]

labels = [np.expand_dims(rgb_to_2D_label(m), axis=2) for m in tqdm(mask_dataset, desc="Converting masks")]
labels = np.array(labels)

print("Unique labels in label dataset are:", np.unique(labels))

# Visual sanity check
import random
image_number = random.randint(0, len(image_dataset) - 1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

# Prepare training and model
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.2, random_state=42)

# FIXED: Replace segmentation_models losses with native TensorFlow losses
weights = [1.0 / n_classes] * n_classes
total_loss = tf.keras.losses.CategoricalCrossentropy()

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', jacard_coef])
model.summary()

history = model.fit(X_train, y_train,
                    batch_size=4,
                    verbose=1,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    shuffle=False)

# IoU Evaluation
from keras.metrics import MeanIoU

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

IOU = MeanIoU(num_classes=n_classes)
IOU.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU.result().numpy())

# ADDED: Additional Metrics Calculation
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

# Flatten the arrays for per-pixel evaluation
y_test_flat = y_test_argmax.flatten()
y_pred_flat = y_pred_argmax.flatten()

# 1. Pixel Accuracy
pixel_accuracy = np.mean(y_test_flat == y_pred_flat)
print(f"Pixel Accuracy: {pixel_accuracy:.4f}")

# 2. Per-class IoU and mean IoU (more detailed)
class_names = ['Urban_land', 'Agriculture_land', 'Rangeland', 'Forest_land', 'Water', 'Barren_land', 'Unknown']
iou_per_class = []
for class_id in range(n_classes):
    intersection = np.sum((y_test_flat == class_id) & (y_pred_flat == class_id))
    union = np.sum((y_test_flat == class_id) | (y_pred_flat == class_id))
    if union > 0:
        iou = intersection / union
    else:
        iou = 0.0
    iou_per_class.append(iou)
    print(f"IoU for {class_names[class_id]}: {iou:.4f}")

mean_iou = np.mean(iou_per_class)
print(f"Mean IoU (calculated): {mean_iou:.4f}")

# 3. Precision, Recall, F1-score
precision, recall, f1, support = precision_recall_fscore_support(y_test_flat, y_pred_flat, average=None, zero_division=0)
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print("\nPer-class Metrics:")
for i in range(n_classes):
    print(f"{class_names[i]}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

print(f"\nMacro-averaged Metrics:")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall: {macro_recall:.4f}")
print(f"F1-score: {macro_f1:.4f}")

# Weighted averages
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_test_flat, y_pred_flat, average='weighted', zero_division=0)
print(f"\nWeighted-averaged Metrics:")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall: {weighted_recall:.4f}")
print(f"F1-score: {weighted_f1:.4f}")

# 4. Confusion Matrix
cm = confusion_matrix(y_test_flat, y_pred_flat)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Classification report (comprehensive summary)
print("\nClassification Report:")
print(classification_report(y_test_flat, y_pred_flat, target_names=class_names, zero_division=0))

# Visual check on prediction
index = random.randint(0, len(X_test) - 1)
pred_img = np.argmax(model.predict(np.expand_dims(X_test[index], 0)), axis=3)[0]

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('Input Image')
plt.imshow(X_test[index])
plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(y_test_argmax[index])
plt.subplot(133)
plt.title('Prediction')
plt.imshow(pred_img)
plt.show()