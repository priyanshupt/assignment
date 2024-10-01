import os
import shutil
from sklearn.model_selection import train_test_split

# Assuming your images and masks are stored in folders
images = [img for img in os.listdir('data/images') if img.endswith('.png')]
masks = [mask for mask in os.listdir('data/masks') if mask.endswith('.png')]

# Split data into 80% train, 20% test
train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

def move_files(file_list, source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file in file_list:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))

move_files(train_imgs, 'data/images', 'data/train/images')
move_files(train_imgs, 'data/masks', 'data/train/masks')
move_files(test_imgs, 'data/images', 'data/test/images')
move_files(test_imgs, 'data/masks', 'data/test/masks')
