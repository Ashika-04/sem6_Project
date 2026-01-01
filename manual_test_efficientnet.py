import torch
from timm import create_model
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["No DR","Mild","Moderate","Severe","Proliferative"]

model = create_model("efficientnetv2_rw_s", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("efficientnet_dr.pth"))
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img_path = input("Enter fundus image path: ").strip().replace('"','')
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img).argmax(1).item()

print("\nPredicted DR Stage:", classes[pred])
