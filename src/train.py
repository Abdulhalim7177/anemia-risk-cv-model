import os
import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. Dataset Loading (Eyes-defy-anemia)
# ==========================================
# We use the local Eyes-defy-anemia dataset (India & Italy folders)
# Labels: Hgb < 11.0 = Anemic (1), Hgb >= 11.0 = Non-Anemic (0)

def load_eyes_defy_anemia(data_dir):
    print("Loading data from", data_dir)
    images = []
    labels = []
    
    regions = ['India', 'Italy']
    
    for region in regions:
        region_path = os.path.join(data_dir, region)
        excel_path = os.path.join(region_path, f"{region}.xlsx")
        
        if not os.path.exists(excel_path):
            continue
            
        df = pd.read_excel(excel_path)
        
        for index, row in df.iterrows():
            subject_num = str(row['Number'])
            hgb = float(row['Hgb'])
            
            # Label: 1 if Hgb < 11.0 (Anemia risk), else 0
            label = 1 if hgb < 11.0 else 0
            
            subject_folder = os.path.join(region_path, subject_num)
            if not os.path.exists(subject_folder):
                continue
                
            # Find the original JPG image
            jpg_files = glob.glob(os.path.join(subject_folder, "*.jpg"))
            if not jpg_files:
                continue
                
            img_path = jpg_files[0]
            img = cv2.imread(img_path)
            
            if img is not None:
                # Basic ROI Crop (crop bottom half where conjunctiva usually is)
                h, w, _ = img.shape
                cropped = img[int(h*0.4):h, :] # take bottom 60%
                
                resized = cv2.resize(cropped, (224, 224))
                normalized = resized / 255.0
                
                images.append(normalized)
                labels.append(label)
                
    print(f"Successfully loaded {len(images)} images.")
    return np.array(images), np.array(labels)

# ==========================================
# 2. Model Architecture
# ==========================================
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

# ==========================================
# 3. Training Loop
# ==========================================
if __name__ == '__main__':
    # Fix random seed for reproducibility
    tf.random.set_seed(42)
    np.random.set_seed(42)

    X, y = load_eyes_defy_anemia('../data/raw/dataset_anemia')
    
    if len(X) == 0:
        print("Error: Could not load any images. Check paths.")
        exit(1)
        
    print(f"Class distribution: {np.bincount(y)}")
    
    # 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Handle Class Imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    print("Class weights applied:", class_weights)

    model = build_model()
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=15, 
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    
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
        print("\nWARNING: Accuracy is below 70%. (This is common with small datasets like Eyes-Defy-Anemia).")
        print("Since this is a fallback validation set, TFLite conversion will still be available.")
    else:
        print("\nModel reached threshold. Ready for TFLite conversion.")
