"""
We compare the two models used to analyze the images.
"""
import os
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt 
import pandas as pd 
import torch.nn as nn
from sklearn.metrics import(accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from torchvision import datasets, transforms, models
from dataset import load_dataset
from features import all_feautures
from config import TEST_DIR, MODEL_PATH,SCALER_PATH,LABELS_NAMES
from preprocess import pre_process_features
from torch.utils.data import DataLoader


CNN_MODEL_PATH = "outputs/cnn_mobilenet_model.pth"

'''
Verify that both the scaler and the SVM model exist before loading them.
'''

def load_svm_model():
    for path in (MODEL_PATH, SCALER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError("File not found")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model correctly Upload")
    return model, scaler 

'''
Verify that the CNN model exists before loading it.
'''
def load_cnn_model():
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError("File not found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
    model.load_state_dict(torch.load(CNN_MODEL_PATH,map_location=device))
    model.to(device)
    model.eval()
    print("CNN Model correctly loaded")
    return model, device

'''
Create a function that computes the metrics based on y_pred and y_true.

The function parameters will include:
- average = "macro", which computes the unweighted mean across the classes.
- zero_division = 0, which prevents errors when a class is missing in the predictions.
'''

def computer_metrics(y_true, y_pred):
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred, average="macro",zero_division=0),
        "recall":recall_score(y_true,y_pred,average="macro",zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)  
        
    }
    
'''
Evaluate the CNN model by resizing the images to 224x224 and normalizing them.
'''

def evaluate_cnn(model,device):
    
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456, 0.406], 
                                                                                                         std=[0.229, 0.224, 0.225])])
    loader = DataLoader(datasets.ImageFolder(TEST_DIR ,transform=transform),  batch_size=8, shuffle=False)
    
    labels, preds = [],[]
    
    with torch.no_grad():
        for images, batch_labels in loader:
            _, predicted = torch.max(model(images.to(device)),1)
            labels.extend(batch_labels.numpy())
            preds.extend(predicted.cpu().numpy())
    return computer_metrics(np.array(labels), np.array(preds))

def evaluate_svm(model,scaler):
    X_test, y_test , _ = load_dataset(TEST_DIR)
    if len(X_test) == 0:
        raise ValueError(" test is empty")
    return computer_metrics(y_test,model.predict(scaler.transform(X_test)))

'''
Create a function that compares the CNN and SVM models based on the evaluation metrics.
'''

def create_comparison_table(svm_results, cnn_results):
    keys = ["accuracy", "precision", "recall","f1"]
    return pd.DataFrame({
        "Metric":["accuracy", "precision", "recall","f1"],
        "SVM":[f"{svm_results[k]:.4f}" for k in keys],
        "CNN":[f"{cnn_results[k]:.4f}"for k in keys],
        
        
    })
    
def plot_comparison_table(df):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center", colWidths=[0.25,0.25,0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    for i in range(len(df.columns)):
        table[(0,i)].set_facecolor("#4CAF50")
        table[(0,i)].set_text_props(weight="bold", color="white")
        
    plt.title("SVM vs CNN")
    output_path = "outputs/comparison_table.png"
    plt.savefig(output_path,dpi=300)
    plt.close()
    print(f"Table saved in {output_path}")

def main():
    print("SVM vs CNN")
    os.makedirs("outputs", exist_ok=True)
    try:
        svm_model, scaler = load_svm_model()
        cnn_model, device = load_cnn_model()
        evaluators = [
            ("SVM", evaluate_svm, (svm_model, scaler)),
            ("CNN", evaluate_cnn, (cnn_model, device)),
        ]
        all_results = {}
        for name, fn, args in evaluators:
            print(f"\n--- Valutation {name} ---")
            all_results[name] = fn(*args)
            print(f"  Accuracy: {all_results[name]['accuracy']:.4f}")
            print(f"  F1-Score: {all_results[name]['f1']:.4f}")

        df = create_comparison_table(all_results["SVM"], all_results["CNN"])
        print("\nTABLE COMPARISATION\n")
        print(df.to_string(index=False))

        plot_comparison_table(df)
        print("\n COMPLETED!")

    except Exception as e:
        print(f"\n Error: {e}")
        print("\nMake sure that the model are trained:")
        print("  - python src/train_svm.py")
        print("  - python src/train_cnn.py")


if __name__ == "__main__":
    main()        
        

    
    
    
