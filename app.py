import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Function to preprocess input images
def preprocess_input_data(image_path, target_size=(150, 150)):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Function to load directories containing images and labels
def load_data_from_directory(directory):
    directories = []
    labels = []
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):  # Check if it's a directory
            directories.append(class_path)
            if class_name == "Non_Demented":
                label = 0  # Assign label 0 for Non_Demented
            else:
                label = 1  # Assign label 1 for Mild_Demented, Moderate_Demented, and Very_Mild_Demented
            labels.extend([label] * len(os.listdir(class_path)))
    return directories, labels

# Load directories from the dataset
directories, labels = load_data_from_directory('E:/projects/alzimer/Dataset')

# Load images from directories
images = []
for directory in directories:
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        if os.path.isfile(image_path):
            images.append(image_path)

# Shuffle data
combined_data = list(zip(images, labels))
np.random.seed(42)  # Ensure reproducibility
np.random.shuffle(combined_data)
images, labels = zip(*combined_data)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess images
train_images = np.array([preprocess_input_data(image) for image in train_images])
val_images = np.array([preprocess_input_data(image) for image in val_images])

# Calculate class weights to handle data imbalance
num_non_demented = np.sum(np.array(train_labels) == 0)
num_demented = np.sum(np.array(train_labels) == 1)
total_samples = len(train_labels)
class_weights = {0: total_samples / (2 * num_non_demented), 1: total_samples / (2 * num_demented)}

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('alzheimer_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_images, np.array(train_labels), epochs=20, batch_size=32, 
                    validation_data=(val_images, np.array(val_labels)),
                    class_weight=class_weights,  # Pass the class weights dictionary here
                    callbacks=[checkpoint, early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(val_images, np.array(val_labels))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
