import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.nn.functional import softmax # Import softmax for confidence scores

save_path = './model/gs_img_weights.pth'

# Training on Gray-Scott images.
# Images of 800x800 pixels.
# Training on a 425x425 crop of the image 100 pixels from the top and 195 pixels from the left

class GSimgCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(GSimgCNN, self).__init__()
        
        # Feature Map Size After 3 Pooling Layers (425 / 8 = 53.125 -> 53)
        final_dim = 53 
        
        # Convolutional Layers (425x425 -> 53x53)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # 212x212
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # 106x106
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2) # 53x53
        )
        
        # Fully Connected Layers
        self.input_features_fc = 128 * final_dim * final_dim 
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.input_features_fc, 512), 
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1) 
        x = self.classifier(x)
        return x

# Training loop
def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs):
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(validation_loader)
        val_epoch_acc = correct_val / total_val
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}')

    return history


# Evaluates the model on the held-out test set
def evaluate_model(model, test_loader, device):
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'\nFinal Accuracy on the TEST set of {total} images: {accuracy * 100:.2f}%')
    return accuracy

# the same evaluation function as before but show the misclassified images
def evaluate_and_report(model, test_loader, test_data, class_to_name, device):
    """
    Evaluates the model and creates a detailed prediction report, 
    including file path, ground truth, prediction, and confidence.
    """
    model.eval() # Set model to evaluation mode
    
    # Lists to store results
    file_paths = []
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    
    total_correct = 0
    total_samples = 0
    
    # Get the list of all file paths corresponding to the test_data
    # test_data.samples contains (file_path, class_index) tuples
    all_test_paths = [path for path, label in test_data.samples]
    
    # Index to track which file path we are on in the all_test_paths list
    path_index = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 1. Get Prediction (index of max logit)
            _, predicted_indices = torch.max(outputs.data, 1)
            
            # 2. Get Confidence (using softmax on raw outputs)
            probabilities = softmax(outputs, dim=1)
            # Find the probability of the predicted class
            predicted_confidence = probabilities.gather(1, predicted_indices.unsqueeze(1)).squeeze(1)
            
            # Store results batch-by-batch
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_indices.cpu().numpy())
            confidence_scores.extend(predicted_confidence.cpu().numpy())

            total_samples += labels.size(0)
            total_correct += (predicted_indices == labels).sum().item()

            # Store file paths for the current batch
            batch_size = labels.size(0)
            file_paths.extend(all_test_paths[path_index : path_index + batch_size])
            path_index += batch_size

    # Convert numerical labels to class names
    true_names = [class_to_name[label] for label in true_labels]
    predicted_names = [class_to_name[label] for label in predicted_labels]

    # Create the DataFrame report
    report_df = pd.DataFrame({
        'File Path': file_paths,
        'Ground Truth': true_names,
        'Prediction': predicted_names,
        'Confidence': [f'{c:.4f}' for c in confidence_scores],
        'Correct': np.array(true_labels) == np.array(predicted_labels)
    })
    
    accuracy = total_correct / total_samples
    print("\n" + "="*50)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_samples})")
    print("="*50 + "\n")

    # --- Print Detailed Report ---
    print("--- Detailed Prediction Report (Test Set) ---")
    
    # Print the full report (may be long)
    print(report_df.to_string()) 
    
    # Optional: Save misclassified images to a CSV
    misclassified = report_df[report_df['Correct'] == False]
    if not misclassified.empty:
        print("\n--- Misclassified Images ---")
        print(misclassified[['File Path', 'Ground Truth', 'Prediction', 'Confidence']].to_string())
        misclassified.to_csv('misclassified_report.csv', index=False)
        print("\nMisclassified images saved to 'misclassified_report.csv'")
        
    return report_df

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
CROP_SIZE = 425 
INPUT_START_X = 195 # 195 pixels from the left
INPUT_START_Y = 100 # 100 pixels from the top

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations with the custom crop
transform = transforms.Compose([
    # ImageFolder yields PIL image, but explicitly converting ensures compatibility
    transforms.Lambda(lambda x: transforms.functional.crop(x, INPUT_START_Y, INPUT_START_X, CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Load datasets
try:
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    validation_data = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_data = datasets.ImageFolder(TEST_DIR, transform=transform) # Load Test Data
except FileNotFoundError as e:
    print(f"\nERROR: Data folder not found. Please verify the paths: {e}")
    exit()

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # Create Test Loader

# Extract the ground truth classes for the evaluation phase
class_to_name = {v: k for k, v in test_data.class_to_idx.items()}

# Train the model
model = GSimgCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = train_model(model, train_loader, validation_loader, criterion, optimizer, NUM_EPOCHS)

# Evaluate and show the mismatches
#evaluate_model(model, test_loader, device)
final_report = evaluate_and_report(model, test_loader, test_data, class_to_name, device)

torch.save(model.state_dict(), save_path)
print("Save model weights to", save_path)
