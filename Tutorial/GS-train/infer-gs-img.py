import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

parser = argparse.ArgumentParser()
parser.add_argument("folder", help="Folder containing images to predict")
parser.add_argument("--model", default="image_model.pth")
parser.add_argument("--num", type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(args.model, map_location=device)
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

folder = Path(args.folder)
image_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg"]:
    image_paths.extend(folder.glob(ext))

if len(image_paths) == 0:
    raise ValueError(f"No images found in {folder}")

selected = random.sample(image_paths, min(args.num, len(image_paths)))

for img_path in selected:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]

    label = idx_to_class[pred]
    confidence = probs[pred].item()

    print(f"{img_path.name}: {label} ({confidence:.3f})")
