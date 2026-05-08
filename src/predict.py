"""
This is the prediction class.

This file is used to apply the trained model to a new image. When we run
the program and pass an unknown image as input, it predicts the result.

It tells us what it predicts, the probability that the image is real or
fake, and whether it is suspicious or authentic.
"""

import numpy as np
import joblib
import sys
from config import MODEL_PATH, SCALER_PATH, LABELS_NAMES
from preprocess import pre_process_features, pre_process
from features import all_feautures

"""
We need to interpret the confidence level of the prediction: if the difference between the two probabilities is small, 
the model is uncertain. Conversely, when one probability is dominant, the model is more confident. 
The threshold indicates how far apart the two probabilities must be in order to consider a decision reliable.
"""
def interpret_confidence(genuine_prob,fake_prob,threshold=0.15):
    medium = 0.35
    diff = abs(genuine_prob - fake_prob)
    if diff < threshold:
        return "LOW"
    elif diff < medium:
        return "MEDIUM"
    else:
        return "HIGHT"
        
        

def predict_image(path_image):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    preprocess = pre_process(path_image)
    img_color, img_gray = pre_process_features(preprocess)
    features = all_feautures(img_color, img_gray)
    features = features.reshape(1,-1)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    prob_genuine = probability[0]
    prob_fake = probability[1]
    label = LABELS_NAMES[pred]
    confidence = interpret_confidence(prob_genuine, prob_fake)
    return label, prob_genuine, prob_fake, confidence

def main():
    if len(sys.argv) < 2 :
        print("usage: python src/predict.py path_image")
        return 
    image_path = sys.argv[1]
    label, prob_genuine, prob_fake, confidence = predict_image(image_path)

    print(f"predict class: {label}")
    print(f"genuine probability: {prob_genuine:.4f}")
    print(f"fake probability: {prob_fake:.4f}")
    print(f"confidence: {confidence}")
    
    if confidence == "LOW":
        print("output: suspect imagine")

    else:
        if label == "fake":
         print("output: fake banknote")
        
        else:
         print("output: genuine banknote")
    
if __name__ == "__main__":
    main()

    

