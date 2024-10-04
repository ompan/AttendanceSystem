import os
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Folders containing images
data_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data'  # Update with the path to your Data folder
augmented_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Augmented Data'  # Update with the path to your Augmented Data folder
resized_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Resized'  # Update with the path to your Resized Data folder

# Load known faces and their encodings
def load_known_faces(*folders):
    known_face_encodings = []
    known_face_names = []

    for folder in folders:
        for file_name in os.listdir(folder):
            if file_name.endswith(('jpg', 'jpeg', 'png')):
                image = face_recognition.load_image_file(os.path.join(folder, file_name))
                encoding = face_recognition.face_encodings(image)
                
                if encoding:  # Check if encoding is found
                    known_face_encodings.append(encoding[0])  # Get the first encoding
                    known_face_names.append(os.path.splitext(file_name)[0])  # Use file name as student name

    return known_face_encodings, known_face_names

# Recognize faces in the uploaded image
def recognize_faces(uploaded_image_path, known_face_encodings, known_face_names):
    uploaded_image = face_recognition.load_image_file(uploaded_image_path)
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the first match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        top, right, bottom, left = face_location
        cv2.rectangle(uploaded_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(uploaded_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the resulting image with recognized faces
    cv2.imshow("Recognized Faces", uploaded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to upload an image
def upload_image():
    uploaded_image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*"))
    )
    if uploaded_image_path:
        recognize_faces(uploaded_image_path, known_face_encodings, known_face_names)

# Example usage
if __name__ == "__main__":
    known_face_encodings, known_face_names = load_known_faces(data_folder, augmented_folder, resized_folder)

    # Create a simple GUI for image upload
    root = tk.Tk()
    root.title("Facial Recognition Attendance System")
    root.geometry("300x200")

    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack(expand=True)

    root.mainloop()
