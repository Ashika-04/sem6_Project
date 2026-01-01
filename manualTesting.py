import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

class DR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = DR_CNN().to(DEVICE)
model.load_state_dict(torch.load("dr_cnn_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img_path = input("Enter fundus image path: ")
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img).argmax(1).item()

print("\nPredicted DR Stage:", classes[pred])
