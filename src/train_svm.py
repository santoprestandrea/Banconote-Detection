"""
This file trains the model.

It loads the training dataset and the testing dataset, standardizes all
the features, and trains the model using the SVM algorithm.

Then, it performs predictions on the test set and prints all the metrics
that help us understand how the model performs, such as accuracy,
precision, and possible margins of error.

Finally, it saves the trained model and applies the scaler.
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from config import TRAIN_DIR, TEST_DIR, MODEL_PATH, SCALER_PATH,LABELS_NAMES
from dataset import load_dataset


'''
Let's create a function that saves the confusion matrix as an image and shows the true and predicted classes,
so that we can visually understand where the model makes mistakes. 
We will save this confusion matrix inside the output path.
'''
def save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("confusion matrix")
    plt.colorbar()
    
    axis_marks = np.arange(len(class_names))
    plt.xticks(axis_marks, class_names)
    plt.yticks(axis_marks, class_names)
    threshold = cm.max() / 2 if cm.size > 0 else 0 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i,j]),
                horizontalalignment = "center",
                color = "white" if cm[i,j] > threshold else "black"  
            )
    plt.ylabel("classe reale")
    plt.xlabel("classe predetta")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi= 300,bbox_inches="tight")
    plt.close()
    
'''
We save a probability chart that will predict for the test every test image we show, 
the probability that it is genuine, the probability that it is fake, 
and we will display it in a pie chart, which we will then save to the output path 
alongside the others.
'''   
def save_probability_chart(y_test, y_pred, y_proba, output_path):
    indices = np.arange(len(y_test))
    plt.figure(figsize=(10,5))
    plt.plot(indices, y_proba[:,0], marker = "o", label = "genuine probability")
    plt.plot(indices, y_proba[:, 1], marker = "s", label ="fake probability")
    for i in range(len(y_test)):
        plt.text(
            i,max(y_proba[i,0],y_proba[i,1]) + 0.02,
            f"R:{LABELS_NAMES[y_test[i]]}\n P:{LABELS_NAMES[y_pred[i]]}",
            ha = "center",
            fontsize = 8
        )
    plt.xlabel("imagine index test")
    plt.ylabel("probability")
    plt.title("probability predict test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
        
    
 
 
"""
Let's save a textual report file containing the best parameters along with the cross-validation score, accuracy, confusion matrix,
and classification report.
We will write all this information into a text file in order to keep track of the program execution.
""" 

def save_result_report(output_path, best_score, accurancy, cm, best_params, class_report):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("output svm training:\n\n " )
        f.write(f"best params founded: {best_params}\n")
        f.write(f"best cross validation score: {best_score:.4}\n")
        f.write(f"accuracy best set: {accurancy:.4}\n")
        f.write(f"confusion matrix: {cm}\n")
        f.write(f"class report: {class_report}\n")
        
 
  

def main():
    print("loading training dataset")
    X_train, y_train, train_path = load_dataset(TRAIN_DIR)
    print("upload testing  dataset")
    X_test, y_test, test_path = load_dataset(TEST_DIR)

    if len(X_train)== 0 or len(X_test) == 0:
        print("dataset is empty  or there are invalid data")
        return 
    print(f"training samples: {len(X_train)}")
    print(f"testing samples: {len(X_test)}")
    print(f"features numbers: {X_train.shape[1]}")

    os.makedirs("models", exist_ok= True)
    os.makedirs("outputs", exist_ok=True)
    
    """
    The param_grid contains all the combinations of parameters that we want to test in order to find the best model.
    The first parameter indicates how much to penalize errors: the higher its value, the less tolerant the model will be.
    The second parameter indicates how much the model focuses on nearby points based on a given scale.
    The third parameter represents the type of kernel to use.
    """
    param_grid = {"C":[1,10,50], "gamma": ["scale", 0.01, 0.001], "kernel": ["rbf"]}
    
    
    '''
    We transform each feature to have a mean value of 0 and a standard deviation of 1,
    because in this way SVM processes the data better, since it is sensitive to scale.
    '''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    '''
    Now, we create the SVM model using SVC, which stands for Support Vector Classifier,
    and it will be used for training.
    '''
    model = SVC(kernel="rbf", C= 10, gamma= "scale", probability= True, random_state= 42)
    print("training SVM")
    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring="f1_macro",cv=3,n_jobs=-1,verbose=1)
    print("reserch with best params with GridSerachCV")
    grid.fit(X_train_scaled,y_train)
    best_model = grid.best_estimator_
    print(best_model)
    print(f"best cross validation score: {grid.best_score_:.4f}")
    model.fit(X_train_scaled, y_train)
    print("prediction's set")
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)
    

    '''
    From now on, we evaluate the model using the confusion matrix,
    accuracy, precision, and recall.
    '''
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"accuracy: {acc:.4f}")
    print("confusion matrix: ")
    print(cm)
    
    print("classification report: ")
    class_report = classification_report(y_test, y_pred, target_names=[LABELS_NAMES[0], LABELS_NAMES[1]])
    print(class_report)

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"model saved in {MODEL_PATH}")
    print(f"scaler saved in {SCALER_PATH}")
    
    """
    Let's save the outputs both in a graphical format and in a textual format.
    """
    save_confusion_matrix(cm=cm, class_names=[LABELS_NAMES[0], LABELS_NAMES[1]], output_path="outputs/confusion_matrix.png")
    save_probability_chart(y_test=y_test, y_pred=y_pred, y_proba=y_proba, output_path="outputs/test_probabilities.png")
    save_result_report(output_path="outputs/results_report.txt",best_score=grid.best_score_,accurancy=acc, cm=cm, best_params=grid.best_params_,class_report=class_report)
   
    print("files output generate:\n- confusion_matrix.png \n- test_probabilies.png \n- results_report.png")
    
if __name__ == "__main__":
    main()





