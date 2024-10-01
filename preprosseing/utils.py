import os
import numpy as np
from PIL import Image
import cv2

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = np.array(img)
        if img is not None:
            images.append(img)
    return np.array(images)

# Function to load both images and masks
def load_data(image_dir, mask_dir):
    X = load_images_from_folder(image_dir)
    y = load_images_from_folder(mask_dir)
    
    # Normalize images
    X = X / 255.0
    y = y / 255.0
    
    # Reshape to add channel dimension
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)
    y = y.reshape(-1, y.shape[1], y.shape[2], 1)
    
    return X, y

# Function to apply CLAHE preprocessing
def apply_clahe_to_images(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_images = []
    
    for img in images:
        img_clahe = clahe.apply(np.array(img, dtype=np.uint8))
        preprocessed_images.append(img_clahe)
    
    return np.array(preprocessed_images)

# Utility to display image
def display_image(image_array):
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.show()
