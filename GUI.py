import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from sklearn.preprocessing import LabelBinarizer
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import csv
import threading
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load pre-trained face recognition model
model = tf.keras.models.load_model(r'C:\Work\Python\AttendanceSystem\custom_face_recognition_model.h5')

# Load label binarizer classes
label_binarizer = LabelBinarizer()
label_binarizer.classes_ = np.load(r'C:\Work\Python\AttendanceSystem\classes.npy', allow_pickle=True)

# Define expected input shape for the model (adjust based on your model)
IMG_SIZE = (224, 224)

# Function to predict face identity
def predict_with_unknown(image):
    try:
        image = cv2.resize(image, IMG_SIZE)  # Resize to model's expected input size
        preprocessed_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        prediction = model.predict(preprocessed_image)
        predicted_index = np.argmax(prediction, axis=1)[0]

        if predicted_index >= len(label_binarizer.classes_):
            return "Unknown"

        return label_binarizer.classes_[predicted_index]
    except Exception as e:
        print(f"Error in predicting face identity: {e}")
        return "Unknown"

# Face Recognition App using Tkinter
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")

        # GUI Components
        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.capture_button = tk.Button(root, text="Capture Image", command=self.capture_image)
        self.capture_button.pack()

        self.attendance_file = r'C:\Work\Python\AttendanceSystem\attendance.csv'

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                image = cv2.imread(file_path)
                if image is None:
                    print("Error: Could not read the image file.")
                    return
                self.process_image(image)
        except Exception as e:
            print(f"Error in uploading image: {e}")

    def capture_image(self):
        try:
            cap = cv2.VideoCapture(0)  # Use the first external camera
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return

            ret, frame = cap.read()
            cap.release()

            if ret:
                self.process_image(frame)
            else:
                print("Error: Could not capture image from camera.")
        except Exception as e:
            print(f"Error in capturing image: {e}")

    def process_image(self, frame):
        try:
            # Convert frame for Tkinter display
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((400, 400))
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image_tk)

            # Run face recognition in a background thread
            threading.Thread(target=self.recognize_face, args=(frame,), daemon=True).start()
        except Exception as e:
            print(f"Error in processing image: {e}")

    def recognize_face(self, frame):
        try:
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print("No face detected.")
                return

            recognized_students = set()
            for (x, y, w, h) in faces:
                face_image = image_np[y:y+h, x:x+w]
                student_name = predict_with_unknown(face_image)
                recognized_students.add(student_name)

                # Draw bounding box and label on the image
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image_np, student_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.save_attendance(recognized_students)

            # Update the displayed image with results
            result_image = Image.fromarray(image_np)
            self.image_tk = ImageTk.PhotoImage(result_image)
            self.image_label.config(image=self.image_tk)
        except Exception as e:
            print(f"Error in recognizing face: {e}")

    def save_attendance(self, recognized_students):
        try:
            # Read existing records to avoid duplicate entries
            existing_records = set()
            if os.path.exists(self.attendance_file):
                with open(self.attendance_file, mode='r', newline='') as file:
                    reader = csv.reader(file)
                    existing_records = {(row[0], row[1]) for row in reader}

            with open(self.attendance_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for student in recognized_students:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if (student, timestamp) not in existing_records:
                        writer.writerow([student, timestamp])
        except Exception as e:
            print(f"Error in saving attendance: {e}")

# Run the Tkinter App
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FaceRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in running the application: {e}")
