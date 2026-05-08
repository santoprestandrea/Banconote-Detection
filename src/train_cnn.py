"""
This file will be responsible for training a deep learning model for banknote recognition, determining whether a banknote 
is genuine or fake.

Using a CNN (Convolutional Neural Network), instead of the classical model that we have already developed,
the model will automatically load the images, resize them because the model requires a standard dimension of 224 x 224, and then
handle the feature extraction process.
"""

import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader 
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

TRAIN_DIR = "data/train" 
TEST_DIR = "data/test"
MODEL_OUTPUT_PATH = "outputs/cnn_mobilenet_model.pth"
BATCH_SIZE = 8 
NUM_EPOCHS = 5
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else"cpu")


transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456, 0.406], 
                                                                                                         std=[0.229, 0.224, 0.225])])

"""
ImageFolder is a function that automatically scans the subfolders and assigns a numerical index to each class in alphabetical 
order.

For example, "fake" will be assigned the index 0 and "genuine" will be assigned the index 1.

Instead of doing this manually, the function handles it automatically.
"""

def load_data():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader

"""
We are going to create the CNN model using MobileNetV2, which is a function that extracts the features,
regularizes them, and maps them into classes.
"""

def create_model(n_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model.to(DEVICE)
    
"""
We train the CNN model on the training set using a number of images that we have specified to be analyzed in parallel.

At each epoch, the model sees the entire dataset and updates its weights to reduce the prediction error.

The loss algorithm is responsible for reducing the error in the predictions.
"""

def train_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p .requires_grad, model.parameters()),lr=LEARNING_RATE)
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images= images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            
    average_loss = total_loss /len(train_loader)
    train_accuracy = correct / total
    
    print(f"Epochs: [{epoch +1}/{NUM_EPOCHS}]")
    print(f"Loss: {average_loss:.4f} ")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    
    
"""
We evaluate the model on the test set to measure the model's generalization capability.

After that, we will calculate the prediction metrics.
"""

def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicter = torch.max(outputs,1)
            """
The .cpu() function moves the tensors from the GPU to the RAM, and the NumPy function
allows us to convert them into NumPy arrays in order to use them with sklearn.
"""
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicter.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names,zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("CNN test results: ")
    print(f"accurancy: {accuracy:.4f}")
    print("report ")
    print(report)
    print("confusion matrix")
    print(cm)
    
    return accuracy, report, cm

def save_model(model):
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict() ,MODEL_OUTPUT_PATH)
    print(f"CNN model save in {MODEL_OUTPUT_PATH}")
    
def main():
    print("CNN training with mobilenetv2")
    print(f"Device Used: {DEVICE}")
    
    train_dataset , test_dataset, train_loader, test_loader = load_data()
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"classes found: {class_names}")
    print(f"Trainig images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)} ")
    model = create_model(num_classes)
    
    train_model(model, train_loader)
    evaluate_model(model, test_loader, class_names)
    save_model(model)
if __name__ == "__main__":
    main()
    
    

    
            
    
        
    