import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = "data"  # Using full dataset for best results
IMG_SIZE = (256, 256)
CLASSES = ["cavity", "filling", "implant", "impacted"]  # Updated classes from our dataset
BATCH_SIZE = 32
EPOCHS = 25  # Reduced for faster training while maintaining quality
VALIDATION_SPLIT = 0.2

def preprocess_image(path):
    """Preprocess a single image"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    # Apply CLAHE for enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize
    img = img / 255.0
    
    return img

def load_dataset():
    """Load and preprocess the entire dataset"""
    data = []
    labels = []
    file_paths = []
    
    print("Loading dataset...")
    for label, class_name in enumerate(CLASSES):
        class_dir = Path(DATA_DIR) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
            
        print(f"Loading {class_name} images...")
        for file in class_dir.glob("*.jpg"):
            try:
                img = preprocess_image(str(file))
                data.append(img)
                labels.append(label)
                file_paths.append(str(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
        
        for file in class_dir.glob("*.png"):
            try:
                img = preprocess_image(str(file))
                data.append(img)
                labels.append(label)
                file_paths.append(str(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    
    print(f"Loaded {len(data)} images total")
    return np.array(data), np.array(labels), file_paths

def create_data_generators():
    """Create data generators with augmentation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

def build_improved_model():
    """Build an improved CNN model"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print detailed metrics"""
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test_classes, y_pred_classes, 
                               target_names=CLASSES))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test_classes, y_pred_classes, CLASSES)
    
    # Calculate and print per-class accuracy
    print("\n" + "="*50)
    print("PER-CLASS ACCURACY")
    print("="*50)
    for i, class_name in enumerate(CLASSES):
        class_mask = y_test_classes == i
        class_accuracy = np.mean(y_pred_classes[class_mask] == i)
        print(f"{class_name:15s}: {class_accuracy:.3f}")

def main():
    """Main training function"""
    print("🦷 Dental X-Ray Classification Model Training")
    print("="*50)
    
    # Load dataset
    X, y, file_paths = load_dataset()
    
    if len(X) == 0:
        print("No images found! Please check your data directory.")
        return
    
    # Reshape for CNN (add channel dimension)
    X = np.expand_dims(X, axis=-1)
    
    # Convert labels to categorical
    y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Create model
    model = build_improved_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save('models/dental_model.h5')
    print("\n✅ Model saved as 'models/dental_model.h5'")
    
    # Save model info
    model_info = {
        'classes': CLASSES,
        'img_size': IMG_SIZE,
        'total_samples': len(X),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'final_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1]
    }
    
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model info saved as 'models/model_info.json'")

if __name__ == "__main__":
    main()