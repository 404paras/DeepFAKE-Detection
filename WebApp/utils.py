import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import cv2
import torch
from torchvision import transforms

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        base_model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=2048,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )

        # Simple linear classifier that outputs directly to num_classes
        self.linear1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size*seq_len, c, h, w)
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        context = lstm_out[:, -1, :]
        return self.linear1(context)

def preprocess_video(video_path, sequence_length=10, im_size=112):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // sequence_length)

    count, extracted = 0, 0
    while cap.isOpened() and extracted < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        if count % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame_rgb))
            extracted += 1
        count += 1

    cap.release()

    if len(frames) < sequence_length:
        for _ in range(sequence_length - len(frames)):
            frames.append(frames[-1])

    video_tensor = torch.stack(frames).unsqueeze(0)
    return video_tensor

def load_model(weights_path="checkpoint.pt"):
    model = DeepfakeDetector(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, video_tensor, device='cpu'):
    video_tensor = video_tensor.to(device)

    print("Running prediction...")
    with torch.no_grad():
        out = model(video_tensor)
        print(f"Model output: {out}")
        pred = torch.argmax(out, dim=1).item()
        print(f"Predicted class: {pred}")
        return "REAL" if pred == 1 else "DEEPFAKE"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model("checkpoint.pt")

    video_path = 'path/to/your/video.mp4'
    video_tensor = preprocess_video(video_path, max_frames=16, frame_sampling='uniform')

    prediction = predict(model, video_tensor, device=device)
    print(f"The video is: {prediction}")

#cd "/Users/I768770/Documents/Mtech 2nd Sem/WebApp" && ./venv/bin/python app.py