import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
IMG_SIZE = (256, 256)
CLASSES = ["cavity", "filling", "implant", "impacted"]
BATCH_SIZE = 32
EPOCHS = 60
VALIDATION_SPLIT = 0.2
SEED = 42

def advanced_preprocess_image(path):
    """Advanced image preprocessing with background segmentation and CLAHE"""
    # Read image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    # Background segmentation to reduce white pixels
    img2 = img - 255
    kernel = np.ones((2, 2))
    kernel2 = np.ones((3, 3))
    
    # Dilate to create mask
    dilated_mask = cv2.dilate(img2, kernel, iterations=3)
    ret, thresh = cv2.threshold(dilated_mask, 0, 255, cv2.THRESH_BINARY)
    dilated_mask2 = cv2.dilate(thresh, kernel2, iterations=3)
    
    # Apply mask to original image
    img = img / 255.0
    res_img = dilated_mask2 * img
    res_img = np.uint8(res_img)
    
    # Apply CLAHE with higher clip limit
    clahe_op = cv2.createCLAHE(clipLimit=20)
    final_img = clahe_op.apply(res_img)
    
    # Resize
    final_img = cv2.resize(final_img, IMG_SIZE)
    
    # Convert to RGB (DenseNet expects 3 channels)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    final_img = final_img / 255.0
    
    return final_img

def load_dataset():
    """Load and preprocess the entire dataset with advanced preprocessing"""
    data = []
    labels = []
    file_paths = []
    
    print("Loading dataset with advanced preprocessing...")
    for label, class_name in enumerate(CLASSES):
        class_dir = Path(DATA_DIR) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
            
        print(f"Loading {class_name} images...")
        for file in class_dir.glob("*.jpg"):
            try:
                img = advanced_preprocess_image(str(file))
                data.append(img)
                labels.append(label)
                file_paths.append(str(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
        
        for file in class_dir.glob("*.png"):
            try:
                img = advanced_preprocess_image(str(file))
                data.append(img)
                labels.append(label)
                file_paths.append(str(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    
    print(f"Loaded {len(data)} images total")
    return np.array(data), np.array(labels), file_paths

def create_data_generators():
    """Create data generators with advanced augmentation"""
    # Advanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        rescale=1./255
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

def build_densenet_model():
    """Build DenseNet201-based model with transfer learning"""
    # Load pre-trained DenseNet201
    base_model = DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Make base model trainable
    base_model.trainable = True
    
    # Create model
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with ELU activation (as in Kaggle notebook)
    x = Dense(256, activation='elu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='elu')(x)
    
    # Output layer
    outputs = Dense(len(CLASSES), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
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
    """Evaluate model and print detailed metrics"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, CLASSES)
    
    return y_pred_classes

def main():
    """Main training function"""
    print("🚀 Starting Advanced Dental X-Ray Classification Training")
    print("="*60)
    
    # Set random seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Load dataset
    X, y, file_paths = load_dataset()
    
    if len(X) == 0:
        print("❌ No images found! Please check your data directory.")
        return
    
    print(f"📊 Dataset loaded: {len(X)} images, {len(CLASSES)} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y
    )
    
    print(f"📈 Training samples: {len(X_train)}")
    print(f"🧪 Test samples: {len(X_test)}")
    
    # Create data generators
    train_datagen, val_datagen = create_data_generators()
    
    # Build model
    print("🏗️ Building DenseNet201 model...")
    model = build_densenet_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📋 Model summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=40,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=15,
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
    print("🎯 Starting training...")
    print(f"⏱️ Training for {EPOCHS} epochs with early stopping")
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/dental_model.h5')
    
    # Save model info
    model_info = {
        'classes': CLASSES,
        'img_size': list(IMG_SIZE),
        'total_samples': len(X),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }
    
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\n🔍 Evaluating model...")
    y_pred_classes = evaluate_model(model, X_test, y_test)
    
    # Print final results
    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETE!")
    print("="*50)
    print(f"📊 Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"💾 Models saved to: models/")
    print(f"📈 Training plots saved to: training_history.png, confusion_matrix.png")
    
    return model, history

if __name__ == "__main__":
    main()