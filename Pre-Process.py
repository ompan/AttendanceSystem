import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Path to the Known Faces folder
known_faces_folder = os.path.expanduser(r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Known_faces')

# Create a new folder for resized images
resized_folder = os.path.join(os.path.dirname(known_faces_folder), 'Resized')
os.makedirs(resized_folder, exist_ok=True)

# Function to resize a single image
def resize_image(file_name):
    try:
        file_path = os.path.join(known_faces_folder, file_name)
        with Image.open(file_path) as img:
            # Resize to 224x224 pixels using LANCZOS filter for high-quality downscaling
            img_resized = img.resize((224, 224), Image.LANCZOS)
            img_resized.save(os.path.join(resized_folder, file_name))
    except Exception:
        pass  # Ignore errors silently

# Get the list of image files
image_files = [file_name for file_name in os.listdir(known_faces_folder) if file_name.lower().endswith(('jpg', 'jpeg', 'png'))]

# Use ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor:
    executor.map(resize_image, image_files)

print("Image resizing completed! Check the 'Resized' folder in the same directory as the Known Faces folder.")
