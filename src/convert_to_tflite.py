import tensorflow as tf
import os

def convert_model():
    if not os.path.exists('../model_output/anemia_model.h5'):
        print("Model not found. Run train.py first.")
        return

    model = tf.keras.models.load_model('../model_output/anemia_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('../model_output/anemia_risk_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Successfully converted model to anemia_risk_model.tflite in model_output/")

if __name__ == '__main__':
    convert_model()
