import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os

def run_evaluation():
    if not os.path.exists('../model_output/anemia_model.h5'):
        print("Model not found. Run train.py first.")
        return

    model = tf.keras.models.load_model('../model_output/anemia_model.h5')

    # Dummy test dataset generation for pipeline validation
    print("Generating synthetic test set for validation...")
    np.random.seed(42)
    X_test = np.random.rand(20, 224, 224, 3)
    y_true = np.random.randint(0, 2, 20)

    preds = model.predict(X_test)
    y_pred = (preds > 0.5).astype(int).flatten()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    print(f"Test Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    RESULT = "PASS" if acc >= 0.70 else "FAIL"
    print(f"RESULT: {RESULT} (threshold: 70% accuracy, published benchmarks: 75-91%)")

if __name__ == '__main__':
    run_evaluation()
