import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import threading
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2  # Import OpenCV

# Folders containing images
known_faces_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Known_faces'  # Path for known faces
augmented_data_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Augmented Data'  # Path for augmented images
resized_data_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Resized'  # Path for resized images

# Load known faces and their corresponding names
known_faces = {}
for filename in os.listdir(known_faces_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the name
        known_faces[name] = os.path.join(known_faces_folder, filename)

# Function to train the model with augmented and resized images
def train_model():
    # Load augmented and resized images for training
    training_data = []
    training_labels = []

    for folder in [augmented_data_folder, resized_data_folder]:
        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the name
                if name in known_faces:  # Ensure name is in known faces
                    img_path = os.path.join(folder, filename)
                    training_data.append(img_path)
                    training_labels.append(name)
    
    # Here you would typically create a training dataset for the model
    # For demonstration, we'll just print out the loaded data
    print(f"Training with {len(training_data)} images.")
    
    return training_data, training_labels

# Recognize faces in the uploaded images
def recognize_faces(uploaded_image_paths, display_label, trained_data):
    results = []
    
    for uploaded_image_path in uploaded_image_paths:
        try:
            print(f"Recognizing faces in {uploaded_image_path}")
            # Read the uploaded image
            image = cv2.imread(uploaded_image_path)

            # Extract faces from the uploaded image
            faces = DeepFace.extract_faces(uploaded_image_path, enforce_detection=False)

            recognized_names = []
            for face_info in faces:
                # Get facial_area coordinates
                if 'facial_area' in face_info:
                    facial_area = face_info['facial_area']
                    x = facial_area['x']
                    y = facial_area['y']
                    w = facial_area['w']
                    h = facial_area['h']
                    
                    # Compare with known faces
                    matched_name = None
                    for name, known_face_path in known_faces.items():
                        result = DeepFace.verify(face_info['face'], known_face_path, enforce_detection=False)
                        if result['verified']:
                            matched_name = name
                            break  # Stop checking once a match is found
                    
                    # Draw rectangle and put name on the image
                    if matched_name:
                        recognized_names.append(matched_name)
                        # Draw rectangle for recognized face
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for recognized
                        cv2.putText(image, matched_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        # Draw rectangle for unknown face
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for unknown
                        cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    print("Facial area not detected for the face.")

            if recognized_names:
                results.append(f"Recognized: {', '.join(recognized_names)}")
            else:
                results.append("No faces recognized.")

            # Display the output image with bounding boxes using Matplotlib
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide the axes
            plt.title("Face Recognition Result")
            plt.show()  # Show the image

        except Exception as e:
            print(f"Error processing {uploaded_image_path}: {e}")
            results.append(f"Error processing {uploaded_image_path}: {str(e)}")
    
    # Display results in the label
    display_results(results, display_label)

# Display results in the label
def display_results(results, display_label):
    result_text = "\n".join(results)
    display_label.config(text=result_text)

# Function to upload images
def upload_images(display_label, trained_data):
    uploaded_image_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*"))
    )
    if uploaded_image_paths:
        print(f"Selected images: {uploaded_image_paths}")
        threading.Thread(target=recognize_faces, args=(uploaded_image_paths, display_label, trained_data)).start()

# Example usage
if __name__ == "__main__":
    # Train the model
    training_data, training_labels = train_model()
    
    # Create a simple GUI for image upload
    root = tk.Tk()
    root.title("Facial Recognition Attendance System")
    root.geometry("600x400")

    upload_button = tk.Button(root, text="Upload Images", command=lambda: upload_images(display_label, training_data))
    upload_button.pack(expand=True)

    # Label to display the results
    display_label = tk.Label(root)
    display_label.pack(expand=True)

    print("Launching GUI...")
    root.mainloop()
