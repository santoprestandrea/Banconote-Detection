"""
In this file we return the metrics of our model based on the test folder.
Therefore, in the main we will load the SVM model, load the saved scaler, and the test images.
Then, features will be extracted using the pipeline in order to compute predictions
and save the resulting metrics.
"""

import numpy as np 
import matplotlib.pyplot as plt
import os 
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from config import TEST_DIR, MODEL_PATH, SCALER_PATH, LABELS_NAMES
from dataset import load_dataset

def save_confusion_matrix(cm, output_path):
    class_names = [LABELS_NAMES[0], LABELS_NAMES[1]]
    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    """
We are extracting the xticks and yticks so that we can include them within the plot.
"""
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    
    """
Now we write the numbers inside the confusion matrix.
"""
    for i in range(cm.shape[0]):
        for j in range(cm.shape[i]):
            plt.text(
                j,i,str(cm[i,j]),
                ha = "center",
                va = "center"
            )
            
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
"""
We save a textual report with all the main metrics such as accuracy, precision,
recall, F1-score, and the confusion matrix. All of this is written to output_path.
"""

def save_report(output_path, accurancy, precision, recall, f1, cm, report):
    
    with open(output_path,"w", encoding="utf-8") as f:
        f.write("Model Evaluation Report")
        f.write(f"accuracy: {accurancy:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall: {recall:.4f}\n")
        f.write(f"f1_score: {f1:.4f}\n")
        f.write("confusion matrix \n")
        f.write(str(cm))
        f.write("classification report")
        f.write(report)
        
def main():
    print("Model Evaluation \n")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model Not Found in {MODEL_PATH} ")
        print("If you have not performed the training, you must first run it with python src/train_svm.py")
        return 
    
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler Not Found in {SCALER_PATH}")
        print("If you have not performed the training, you must first run it with python src/train_svm.py")
        return 
    
    """
Now we load the model and the scaler.
"""

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print("Model and Scaler loaded correctly")
    
    print("loading text images")
    X_test, y_test, test_paths = load_dataset(TEST_DIR)
    
    if len(X_test) == 0:
        print("there are not imagines")
        return
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    accurancy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro" )
    recall = recall_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    report = classification_report(y_test, y_pred, labels=[0,1], target_names=[LABELS_NAMES[0], LABELS_NAMES[1]])
    
    os.makedirs("outputs", exist_ok=True)
    
    save_confusion_matrix(cm,"outputs/evaluation_confusion_matrix.png")
    save_report("outputs/evaluation_report.txt", accurancy, precision, recall, f1, cm, report)
    
    print("output")
    print(f"accuracy: {accurancy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1_score: {f1:.4f}")
    print("confusion matrix")
    print(cm)
    print("report")
    print(report)

    print("genereted files ")
    print("- outputs/evaluation_confusion_matrix.png")
    print("- outputs/evaluation_report.txt")
    
if __name__ == "__main__":
    main()
