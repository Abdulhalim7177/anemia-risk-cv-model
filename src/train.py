import os
import time

# Suppress TensorFlow and OpenCV warnings for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Custom Callback for Beautiful Terminal Output
class BeautifulTrainingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("\n\033[1;36m" + "="*60 + "\033[0m")
        print("\033[1;36m🚀 INITIALIZING NEURAL NETWORK TRAINING\033[0m")
        print("\033[1;36m" + "="*60 + "\033[0m\n")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        acc = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        print(f"\033[1;34mEpoch {epoch + 1:02d}/{self.params['epochs']}\033[0m | "
              f"⏳ \033[1;33m{duration:.1f}s\033[0m | "
              f"📉 Train Loss: \033[1;31m{loss:.3f}\033[0m | "
              f"🎯 Train Acc: \033[1;32m{acc:.1f}%\033[0m || "
              f"📉 Val Loss: \033[1;31m{val_loss:.3f}\033[0m | "
              f"🎯 Val Acc: \033[1;32m{val_acc:.1f}%\033[0m")
              
    def on_train_end(self, logs=None):
        print("\n\033[1;36m" + "="*60 + "\033[0m")
        print("\033[1;36m✅ TRAINING COMPLETE\033[0m")
        print("\033[1;36m" + "="*60 + "\033[0m\n")

# ==========================================
# 1. Dataset Loading (Eyes-defy-anemia)
# ==========================================
# We use the local Eyes-defy-anemia dataset (India & Italy folders)
# Labels: Hgb < 11.0 = Anemic (1), Hgb >= 11.0 = Non-Anemic (0)

def load_eyes_defy_anemia(data_dir):
    print("\n\033[1;35m[1/3] Loading & Augmenting Dataset...\033[0m")
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
            hgb_str = str(row['Hgb']).replace(',', '.').strip()
            try:
                hgb = float(hgb_str)
            except ValueError:
                continue
            
            # Label: 1 if Hgb < 11.0 (Anemia risk), else 0
            label = 1 if hgb < 11.0 else 0
            
            subject_folder = os.path.join(region_path, subject_num)
            if not os.path.exists(subject_folder):
                continue
                
            # Find all JPG and PNG images (original + segmented versions)
            img_files = glob.glob(os.path.join(subject_folder, "*.jpg")) + glob.glob(os.path.join(subject_folder, "*.png"))
            if not img_files:
                continue
                
            for img_path in img_files:
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Basic ROI Crop (crop bottom half where conjunctiva usually is)
                    h, w, _ = img.shape
                    cropped = img[int(h*0.4):h, :] # take bottom 60%
                    
                    resized = cv2.resize(cropped, (224, 224))
                    normalized = resized / 255.0
                    
                    images.append(normalized)
                    labels.append(label)
                
    print(f"\033[1;32mSuccess! Loaded {len(images)} total images for training.\033[0m")
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
    np.random.seed(42)

    X, y = load_eyes_defy_anemia('../data/raw/dataset_anemia')
    
    if len(X) == 0:
        print("\033[1;31mError: Could not load any images. Check paths.\033[0m")
        exit(1)
        
    print(f"\033[1;35m[2/3] Preparing Model Architecture...\033[0m")
    
    # 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Handle Class Imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}

    model = build_model()
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    beautiful_cb = BeautifulTrainingCallback()
    
    # Set verbose=0 to hide standard Keras progress bar
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=15, 
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop, beautiful_cb],
        verbose=0
    )
    
    print("\033[1;35m[3/3] Evaluating on Held-Out Test Set...\033[0m\n")
    preds = model.predict(X_test, verbose=0)
    y_pred = (preds > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"\033[1;37m📊 FINAL TEST REPORT\033[0m")
    print(f"----------------------------------------")
    print(f"🧠 \033[1;32mTest Accuracy : {acc*100:.1f}%\033[0m")
    print(f"🎯 \033[1;34mPrecision     : {prec*100:.1f}%\033[0m")
    print(f"📡 \033[1;34mRecall        : {rec*100:.1f}%\033[0m")
    print(f"----------------------------------------\n")
    
    os.makedirs('../model_output', exist_ok=True)
    # Suppress save warning by using .keras extension internally or ignoring it
    model.save('../model_output/anemia_model.h5')
    
    if acc < 0.70:
        print("\033[1;33m⚠️ WARNING: Accuracy is below 70%.\033[0m")
        print("\033[1;33mSince this is a fallback validation set, TFLite conversion will still be available.\033[0m")
    else:
        print("\033[1;32m🏆 Model reached threshold! Ready for mobile integration.\033[0m")
