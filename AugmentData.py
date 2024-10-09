import os
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Path to the Known Faces folder
known_faces_folder = os.path.expanduser(r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Known_faces')

# Path to the Augmented Data folder in OneDrive
augmented_folder = os.path.join(os.path.dirname(known_faces_folder), 'Augmented Data')

# Create the Augmented Data folder if it doesn't exist
os.makedirs(augmented_folder, exist_ok=True)

# Elastic transformation function
def elastic_transform(image, alpha, sigma):
    image_np = np.array(image)
    
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    
    random_state = np.random.RandomState(None)
    shape = image_np.shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    transformed_image = np.zeros_like(image_np)
    
    for i in range(image_np.shape[-1]):
        transformed_image[..., i] = map_coordinates(image_np[..., i], indices, order=1, mode="reflect").reshape(shape)
    
    if transformed_image.shape[-1] == 1:
        transformed_image = transformed_image.squeeze(-1)
    
    return Image.fromarray(transformed_image.astype(np.uint8))

# Function to perform multiple augmentations
def augment_image(image_path):
    with Image.open(image_path) as img:
        # Various augmentation techniques
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))  # Blurring
        img_array = np.array(img)
        
        # Noise Addition
        noise = np.random.normal(0, 25, img_array.shape)
        img_noisy = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
        
        # Elastic Transformation
        img_elastic = elastic_transform(img, alpha=36, sigma=5)
        
        # Grayscale Conversion
        img_gray = img.convert('L')
        
        # Darkening
        img_dark = ImageEnhance.Brightness(img).enhance(0.3)  # Darken by 70%

        # Zoom Out (Far Distance)
        img_zoom_out = Image.new('RGB', (img.width, img.height), (255, 255, 255))  # White canvas
        shrink_ratio = random.uniform(0.5, 0.8)  # Shrink to 50-80% of original size
        img_shrunk = img.resize((int(img.width * shrink_ratio), int(img.height * shrink_ratio)))
        offset_x = (img.width - img_shrunk.width) // 2
        offset_y = (img.height - img_shrunk.height) // 2
        img_zoom_out.paste(img_shrunk, (offset_x, offset_y))  # Center the shrunk image

        # Save augmented images
        img_flipped.save(os.path.join(augmented_folder, f'flipped_{os.path.basename(image_path)}'))
        img_blurred.save(os.path.join(augmented_folder, f'blurred_{os.path.basename(image_path)}'))
        img_noisy.save(os.path.join(augmented_folder, f'noisy_{os.path.basename(image_path)}'))
        img_elastic.save(os.path.join(augmented_folder, f'elastic_{os.path.basename(image_path)}'))
        img_gray.save(os.path.join(augmented_folder, f'grayscale_{os.path.basename(image_path)}'))
        img_dark.save(os.path.join(augmented_folder, f'dark_{os.path.basename(image_path)}'))
        img_zoom_out.save(os.path.join(augmented_folder, f'zoom_out_{os.path.basename(image_path)}'))

# Process each image in the Known Faces folder
for file_name in os.listdir(known_faces_folder):
    if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
        augment_image(os.path.join(known_faces_folder, file_name))

print("Data augmentation completed! Check the 'Augmented Data' folder in OneDrive.")
