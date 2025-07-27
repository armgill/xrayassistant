import os
import json
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
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ContinuousLearner:
    def __init__(self):
        self.feedback_file = 'feedback_data.json'
        self.model_path = "models/best_model.h5"
        self.classes = ["cavity", "filling", "implant", "impacted"]
        self.model = None
        self.feedback_data = []
        self.load_feedback_data()
    
    def load_feedback_data(self):
        """Load feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
                print(f"📊 Loaded {len(self.feedback_data)} feedback entries")
            else:
                print("📝 No feedback data found")
                self.feedback_data = []
        except Exception as e:
            print(f"❌ Error loading feedback: {e}")
            self.feedback_data = []
    
    def advanced_preprocess_image(self, image_path):
        """Advanced image preprocessing with background segmentation and CLAHE"""
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
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
        final_img = cv2.resize(final_img, (256, 256))
        
        # Convert to RGB (DenseNet expects 3 channels)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        final_img = final_img / 255.0
        
        return final_img
    
    def prepare_feedback_dataset(self):
        """Prepare dataset from feedback data"""
        print("🔄 Preparing feedback dataset...")
        
        data = []
        labels = []
        valid_feedback = []
        
        for feedback in self.feedback_data:
            try:
                # Try to find the image in various locations
                image_path = None
                possible_paths = [
                    feedback['image_path'],
                    f"uploads/{feedback['image_path']}",
                    f"data/{feedback['image_path']}",
                    f"temp/{feedback['image_path']}"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        break
                
                if image_path is None:
                    print(f"⚠️ Image not found: {feedback['image_path']}")
                    continue
                
                # Preprocess image
                img = self.advanced_preprocess_image(image_path)
                
                # Use user correction as the true label
                true_label = feedback['user_correction']
                
                data.append(img)
                labels.append(true_label)
                valid_feedback.append(feedback)
                
            except Exception as e:
                print(f"❌ Error processing feedback: {e}")
                continue
        
        print(f"✅ Prepared {len(data)} images from feedback")
        return np.array(data), np.array(labels), valid_feedback
    
    def build_densenet_model(self):
        """Build DenseNet201-based model with transfer learning"""
        # Load pre-trained DenseNet201
        base_model = DenseNet201(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
        )
        
        # Make base model trainable
        base_model.trainable = True
        
        # Create model
        inputs = Input(shape=(256, 256, 3))
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers with ELU activation
        x = Dense(256, activation='elu')(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='elu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='elu')(x)
        
        # Output layer
        outputs = Dense(len(self.classes), activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def load_existing_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print("✅ Loaded existing model")
                return True
            else:
                print("📝 No existing model found, will create new one")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def retrain_with_feedback(self, epochs=10, learning_rate=0.0001):
        """Retrain model with feedback data"""
        print("🚀 Starting continuous learning...")
        
        # Prepare feedback dataset
        X_feedback, y_feedback, valid_feedback = self.prepare_feedback_dataset()
        
        if len(X_feedback) == 0:
            print("❌ No valid feedback data to train with")
            return False
        
        # Load or create model
        if not self.load_existing_model():
            print("🏗️ Creating new model...")
            self.model = self.build_densenet_model()
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Split feedback data
        X_train, X_val, y_train, y_val = train_test_split(
            X_feedback, y_feedback, test_size=0.2, random_state=42, stratify=y_feedback
        )
        
        print(f"📊 Training with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print("🎯 Training model with feedback...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("🔍 Evaluating model...")
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print results
        print("\n" + "="*50)
        print("📊 CONTINUOUS LEARNING RESULTS")
        print("="*50)
        print(f"📈 Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"📈 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=self.classes))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        print(f"💾 Model saved to {self.model_path}")
        
        # Save training info
        training_info = {
            'feedback_samples': len(X_feedback),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['accuracy']),
            'feedback_used': len(valid_feedback)
        }
        
        with open('models/continuous_learning_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print("✅ Continuous learning complete!")
        return True
    
    def analyze_feedback_trends(self):
        """Analyze feedback trends and model performance"""
        if not self.feedback_data:
            print("📝 No feedback data to analyze")
            return
        
        print("\n📊 FEEDBACK ANALYSIS")
        print("="*30)
        
        # Basic statistics
        total_feedback = len(self.feedback_data)
        correct_predictions = sum(1 for f in self.feedback_data if f['correct'])
        accuracy = correct_predictions / total_feedback
        
        print(f"📈 Total Feedback: {total_feedback}")
        print(f"✅ Correct Predictions: {correct_predictions}")
        print(f"❌ Incorrect Predictions: {total_feedback - correct_predictions}")
        print(f"🎯 Overall Accuracy: {accuracy*100:.1f}%")
        
        # Class distribution
        print("\n🦷 Class Distribution in Feedback:")
        class_counts = {}
        for feedback in self.feedback_data:
            class_name = self.classes[feedback['user_correction']]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name.title()}: {count}")
        
        # Confidence analysis
        confidences = [f['confidence'] for f in self.feedback_data]
        print(f"\n🎯 Average AI Confidence: {np.mean(confidences)*100:.1f}%")
        print(f"🎯 Confidence Range: {np.min(confidences)*100:.1f}% - {np.max(confidences)*100:.1f}%")
        
        # Recent performance
        recent_feedback = self.feedback_data[-10:] if len(self.feedback_data) >= 10 else self.feedback_data
        recent_correct = sum(1 for f in recent_feedback if f['correct'])
        recent_accuracy = recent_correct / len(recent_feedback) if recent_feedback else 0
        
        print(f"\n📈 Recent Performance (Last {len(recent_feedback)} predictions): {recent_accuracy*100:.1f}%")

def main():
    """Main function for continuous learning"""
    print("🔄 Dental X-Ray AI - Continuous Learning System")
    print("="*50)
    
    learner = ContinuousLearner()
    
    # Analyze current feedback
    learner.analyze_feedback_trends()
    
    # Check if we have enough feedback to retrain
    if len(learner.feedback_data) < 5:
        print(f"\n⚠️ Not enough feedback data ({len(learner.feedback_data)} samples)")
        print("📝 Need at least 5 feedback samples to retrain")
        print("💡 Use the web app to provide more feedback")
        return
    
    # Ask user if they want to retrain
    print(f"\n🤔 Found {len(learner.feedback_data)} feedback samples")
    print("Would you like to retrain the model with this feedback?")
    
    # For now, auto-retrain if we have enough data
    if len(learner.feedback_data) >= 5:
        print("🚀 Auto-retraining with available feedback...")
        success = learner.retrain_with_feedback(epochs=10, learning_rate=0.0001)
        
        if success:
            print("\n🎉 Model successfully updated with your feedback!")
            print("🔄 The web app will now use the improved model")
        else:
            print("\n❌ Failed to retrain model")
    else:
        print("📝 Need more feedback to retrain effectively")

if __name__ == "__main__":
    main() 