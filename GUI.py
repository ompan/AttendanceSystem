import tkinter as tk
from tkinter import filedialog, Label
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from custom_face_recognition_model.h5 import predict_with_unknown  # type: ignore # Make sure to replace with your actual model script name

# Create a Tkinter window
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")

        # Create a label to display the uploaded image
        self.image_label = Label(root)
        self.image_label.pack()

        # Create a label to display the recognized name
        self.result_label = Label(root, text="Recognized: ", font=("Helvetica", 16))
        self.result_label.pack()

        # Create a button to upload an image
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

    def upload_image(self):
        # Open a file dialog to choose an image
        file_path = filedialog.askopenfilename()
        if file_path:
            self.show_image(file_path)
            self.recognize_face(file_path)

    def show_image(self, file_path):
        # Load the image using OpenCV
        image = cv2.imread(file_path)
        # Convert the image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to PIL format and resize
        image = Image.fromarray(image)
        image = image.resize((400, 400), Image.ANTIALIAS)
        # Create an ImageTk object and display it in the label
        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk)
        self.image_label.image = self.image_tk  # Keep a reference

    def recognize_face(self, file_path):
        # Load the image and predict the face
        image = cv2.imread(file_path)
        student_name = predict_with_unknown(image)
        # Display the recognized name or "Unknown"
        self.result_label.config(text=f"Recognized: {student_name}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
