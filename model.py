import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.preprocessing import LabelBinarizer

# Load your model
model = tf.keras.models.load_model('C:\Work\Python\AttendanceSystem\custom_face_recognition_model.h5')

# Load your label binarizer classes (update with your actual classes)
label_binarizer = LabelBinarizer()
label_binarizer.classes_ = np.load('classes.npy', allow_pickle=True)  # Load your class names if saved

# Function to predict with unknown face handling
def predict_with_unknown(image):
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    prediction = model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Handle unknown face
    if predicted_index >= len(label_binarizer.classes_):
        return "Unknown"

    return label_binarizer.classes_[predicted_index]  # Return the predicted name
