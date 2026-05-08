"""
This file builds the dataset that will be used by the model.

For each image, it reads the file, applies preprocessing, and extracts
the features. It then adds the feature vector to X, adds the label to Y,
and saves the file path.

Finally, it returns a matrix of features and a vector of labels.

The variables X and Y will be used for the model preprocessing.
"""

import numpy as np
import cv2
import os 
from config import LABELS
from preprocess import pre_process_features, pre_process
from features import all_feautures

VALID_EXTENTIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp") 

def load_dataset(director_base):
    X = []
    y = []
    path = []

    for class_name, label in LABELS.items():
        class_dir = os.path.join(director_base, class_name)
        
        if not os.path.isdir(class_dir):
            print("directory dosen't exist")
            continue
        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith(VALID_EXTENTIONS):
                continue
            img_path = os.path.join(class_dir,file_name)

            try:
                preprocessed = pre_process(img_path)
                color, img_gray = pre_process_features(preprocessed)
                features = all_feautures(color, img_gray)
                print(f"{img_path} -> lenght feature: {len(features)}")
                X.append(features)
                y.append(label)
                path.append(img_path)

            except Exception as e:
                print(f"error: {e}")

    if len(X) == 0:
        return np.array([]), np.array([]), path
    
    features_lenght = [len(f) for f in X]
    unique_lenght = set(features_lenght)
    if len(unique_lenght)!= 1:
        raise ValueError("feature can't have different size")
    return np.vstack(X), np.array(y), path
    
        




