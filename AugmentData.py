import os
from PIL import Image, ImageEnhance
import random

# Path to the OneDrive temporary photos folder
onedrive_temp_folder = os.path.expanduser(r'C:\Users\vidit\OneDrive - Manipal University Jaipur')

# Path to the Augmented Data folder in OneDrive
augmented_folder = os.path.join(onedrive_temp_folder, 'Augmented Data')

# Create the Augmented Data folder if it doesn't exist
os.makedirs(augmented_folder, exist_ok=True)

def augment_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Scaling: Resize image to 80% and 120%
        img_scaled_small = img.resize((int(img.width * 0.8), int(img.height * 0.8)))
        img_scaled_large = img.resize((int(img.width * 1.2), int(img.height * 1.2)))
        
        # Cropping: Random crop of 80% of the original image size
        left = random.randint(0, int(img.width * 0.2))
        top = random.randint(0, int(img.height * 0.2))
        right = img.width - random.randint(0, int(img.width * 0.2))
        bottom = img.height - random.randint(0, int(img.height * 0.2))
        img_cropped = img.crop((left, top, right, bottom))
        
        # Color Variation: Change brightness and contrast
        img_bright = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))  # Random brightness
        img_contrast = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))  # Random contrast

        # Save augmented images in the Augmented Data folder
        img_scaled_small.save(os.path.join(augmented_folder, f'scaled_small_{os.path.basename(image_path)}'))
        img_scaled_large.save(os.path.join(augmented_folder, f'scaled_large_{os.path.basename(image_path)}'))
        img_cropped.save(os.path.join(augmented_folder, f'cropped_{os.path.basename(image_path)}'))
        img_bright.save(os.path.join(augmented_folder, f'brightened_{os.path.basename(image_path)}'))
        img_contrast.save(os.path.join(augmented_folder, f'contrasted_{os.path.basename(image_path)}'))

# Process each image in the temporary folder
for file_name in os.listdir(onedrive_temp_folder):
    if file_name.endswith(('jpg', 'jpeg', 'png')):
        augment_image(os.path.join(onedrive_temp_folder, file_name))

print("Data augmentation completed! Check the 'Augmented Data' folder in OneDrive.")
