import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.preprocessing import LabelBinarizer
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load your model
model = tf.keras.models.load_model(r'C:\Work\Python\AttendanceSystem\custom_face_recognition_model.h5')

# Load your label binarizer classes (update with your actual classes)
label_binarizer = LabelBinarizer()
label_binarizer.classes_ = np.load(r'C:\Work\Python\AttendanceSystem\classes.npy', allow_pickle=True)

# Function to predict with unknown face handling
def predict_with_unknown(image):
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    prediction = model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]

    if predicted_index >= len(label_binarizer.classes_):
        return "Unknown", None

    return label_binarizer.classes_[predicted_index], None

# Create a Tkinter window
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((400, 400))
        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk)

        # Recognize faces in the image
        self.recognize_face(file_path)

    def recognize_face(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image_np = np.array(image)

        # Detect faces using OpenCV (you may want to update the face detection method as needed)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_image = image_np[y:y+h, x:x+w]
            student_name, _ = predict_with_unknown(face_image)

            # Draw bounding box and label on the image
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_np, student_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the image to PIL format and display it
        result_image = Image.fromarray(image_np)
        self.image_tk = ImageTk.PhotoImage(result_image)
        self.image_label.config(image=self.image_tk)

# Create and run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
