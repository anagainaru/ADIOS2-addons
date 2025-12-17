import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os

model_path = './model/gs_img_weights.pth'

# Classify new Gray-Scott images.
# Images of 800x800 pixels.
# Training was done on a 425x425 crop of the image 100 pixels from the top and 195 pixels from the left

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

def classfiy_images(model, new_loader, device):
    model.eval()
    with torch.no_grad():
        for images, filenames in new_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for idx in range(len(filenames)):
                print(f"{filenames[idx]} {predicted[idx]}")

class InferenceImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Use glob to find all common image file types (you might need to adjust this list)
        # It creates a list of full paths to all image files.
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for pattern in image_patterns:
            self.image_files.extend(glob.glob(os.path.join(self.root_dir, pattern)))
        
        # Store just the base filename for identification (optional, but helpful)
        self.filenames = [os.path.basename(f) for f in self.image_files]

    def __len__(self):
        # The total number of images in the folder
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Get the image path
        img_path = self.image_files[idx]
        
        # 2. Load the image using PIL
        image = Image.open(img_path).convert('RGB') # Convert to RGB if needed
        
        # 3. Apply the transformation
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image # If no transform, return the PIL image
            
        # We return the tensor and the original filename/path for identification
        return image_tensor, self.filenames[idx]


INPUT_DIR = 'data/new'
CROP_SIZE = 425 
INPUT_START_X = 195 # 195 pixels from the left
INPUT_START_Y = 100 # 100 pixels from the top
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations with the custom crop
transform = transforms.Compose([
    # ImageFolder yields PIL image, but explicitly converting ensures compatibility
    transforms.Lambda(lambda x: transforms.functional.crop(x, INPUT_START_Y, INPUT_START_X, CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

new_data = InferenceImageFolder(INPUT_DIR, transform=transform)
new_loader = DataLoader(new_data, batch_size=BATCH_SIZE, shuffle=False)

model = GSimgCNN().to(device)
model.load_state_dict(torch.load(model_path))
classfiy_images(model, new_loader, device)
