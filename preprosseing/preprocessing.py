import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def augment_data(X, y):
    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    return datagen.flow(X, y, batch_size=32)
