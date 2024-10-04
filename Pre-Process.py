from PIL import Image
import os

# Path to the OneDrive Additional Data folder
additional_data_folder = os.path.expanduser(r'C:\Users\vidit\OneDrive - Manipal University Jaipur')

# Create a new folder for resized images
resized_folder = os.path.join(additional_data_folder, 'Resized')
os.makedirs(resized_folder, exist_ok=True)

# Resize images
for file_name in os.listdir(additional_data_folder):
    if file_name.endswith(('jpg', 'jpeg', 'png')):
        with Image.open(os.path.join(additional_data_folder, file_name)) as img:
            img_resized = img.resize((224, 224))  # Resize to 224x224 pixels
            img_resized.save(os.path.join(resized_folder, file_name))

print("Image resizing completed! Check the 'Resized' folder in Additional Data.")
