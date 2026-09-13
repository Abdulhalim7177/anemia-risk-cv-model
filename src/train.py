import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# 1. Dataset Instructions (Kirpalsingh225)
# ==========================================
# Link: https://github.com/kirpalsingh225/Detection-of-Anemia-Using-Conjuctiva-Images
# To download: 
# git clone https://github.com/kirpalsingh225/Detection-of-Anemia-Using-Conjuctiva-Images.git
# and place the images in ../data/raw/kirpalsingh225/

# ==========================================
# 2. ROI Cropping (Preprocessing)
# ==========================================
def crop_conjunctiva_roi(image_path, output_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((224, 224, 3))
    # In a real pipeline, apply segmentation here if masks are available
    resized = cv2.resize(img, output_size)
    return resized / 255.0  # Normalize to [0, 1]

# ==========================================
# 3. Data Loading & Splitting
# ==========================================
def load_data(data_dir):
    # Dummy data loader if real dataset is not found (for CI/CD initialization)
    print("Loading data from", data_dir)
    images = []
    labels = []
    
    # We will generate synthetic data for the sake of pipeline initialization if empty
    # so that the user can push to git and verify the pipeline works.
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("WARNING: Dataset not found. Generating synthetic data to test pipeline.")
        for _ in range(100):
            images.append(np.random.rand(224, 224, 3))
            labels.append(np.random.randint(0, 2))
    else:
        # Actual loading logic would go here:
        # for label in ['anemic', 'non_anemic']:
        #    ... load crop_conjunctiva_roi(path) ...
        pass
        
    return np.array(images), np.array(labels)

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Fix random seed
    tf.random.set_seed(42)
    np.random.set_seed(42)

    X, y = load_data('../data/raw/kirpalsingh225')
    
    # 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = build_model()
    
    # Fast training just for pipeline validation
    print("Starting training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=16)
    
    print("\nEvaluating on Test Set (15% held-out)...")
    preds = model.predict(X_test)
    y_pred = (preds > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"Final Test Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    
    os.makedirs('../model_output', exist_ok=True)
    model.save('../model_output/anemia_model.h5')
    
    if acc < 0.70:
        print("\nWARNING: Accuracy is below 70%. DO NOT proceed to TFLite conversion.")
        print("Suggested adjustments:")
        print("1. Increase epochs (e.g., 15-30).")
        print("2. Add data augmentation (rotation, flipping, brightness).")
        print("3. Check class balance in the dataset.")
    else:
        print("\nModel reached threshold. Ready for TFLite conversion.")
