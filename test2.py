import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Set the default encoding to UTF-8 to avoid charmap issues
import sys
if sys.platform.startswith('win'):
    import io
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths to the folders
known_faces_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Known_faces'  # Known Faces folder
resized_data_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Resized'  # Resized images folder
augmented_data_folder = r'C:\Users\vidit\OneDrive - Manipal University Jaipur\Data\Augmented Data'  # Augmented images folder

# Hyperparameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50

# Step 1: Load Data from a given folder
def load_images_and_labels(data_folder):
    images = []
    labels = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(data_folder, filename)
            image = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)

            # The label is the filename without extension (assuming filename is the name)
            label = os.path.splitext(filename)[0]
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load images and labels from all relevant folders
known_faces_images, known_faces_labels = load_images_and_labels(known_faces_folder)
resized_images, resized_labels = load_images_and_labels(resized_data_folder)
augmented_images, augmented_labels = load_images_and_labels(augmented_data_folder)

# Combine images and labels from all sources
images = np.concatenate((known_faces_images, resized_images, augmented_images), axis=0)
labels = np.concatenate((known_faces_labels, resized_labels, augmented_labels), axis=0)

# Normalize the pixel values
images = images / 255.0

# Encode labels to categorical values
label_binarizer = LabelBinarizer()
labels_encoded = label_binarizer.fit_transform(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Step 2: Define the CNN Model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_binarizer.classes_), activation='softmax')  # Output layer based on the number of students
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = build_model()

# Step 3: Train the Model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Step 4: Save the Model
model.save('custom_face_recognition_model.h5')

# Step 5: Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Function to predict with unknown face handling
def predict_with_unknown(image):
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Ensure normalization
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    prediction = model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Check if predicted index is within the bounds
    if predicted_index >= len(label_binarizer.classes_):
        return "Unknown"  # Handle unknown face

    return label_binarizer.classes_[predicted_index]  # Return the predicted student's name

# Safely print any Unicode messages
def print_safe(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('utf-8').decode('utf-8', 'ignore'))

# Example of printing training completion message
print_safe("Model training completed successfully.")
