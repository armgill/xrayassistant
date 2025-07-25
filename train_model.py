import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


DATA_DIR = "data"
IMG_SIZE = (256, 256)
CLASSES = ["cavity", "crown", "filling", "normal"]


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize
    return img


data = []
labels = []

for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        try:
            img = preprocess_image(file_path)
            data.append(img)
            labels.append(label)
        except:
            print(f"Failed to load {file_path}")

X = np.expand_dims(np.array(data), axis=-1)
y = tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("dental_model.h5")
print("Model saved as dental_model.h5")