from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import cv2
import os
import torch.nn as nn
import torchvision.models as models

# ----------------- CNN + LSTM Model -----------------
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.lstm = nn.LSTM(2048, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        with torch.no_grad():
            features = self.cnn(x).squeeze()
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1])

# ----------------- Flask App -----------------
app = Flask(__name__)
CORS(app)  # allow frontend at localhost:3000 to access backend

# ----------------- Load Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "cnn_lstm_action1.pth"  # make sure this file is in backend folder

checkpoint = torch.load(model_path, map_location=device)
CLASSES = checkpoint['classes']  # list of classes
model = CNN_LSTM(num_classes=len(CLASSES)).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ----------------- Video Preprocessing -----------------
frames_per_clip = 16
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frames_per_clip:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
    cap.release()
    
    # If video is too short, duplicate last frame
    if len(frames) == 0:
        raise ValueError("Video has no frames")
    while len(frames) < frames_per_clip:
        frames.append(frames[-1])
        
    return torch.stack(frames).unsqueeze(0)  # [1, T, C, H, W]

# ----------------- API Route -----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    print(f"Received file: {file_path}")  # DEBUG

    # Check if video opens
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 400
    cap.release()

    try:
        video_tensor = preprocess_video(file_path).to(device)
        with torch.no_grad():
            outputs = model(video_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
        print(f"Prediction index: {pred_idx}")  # DEBUG
        return jsonify({'prediction': CLASSES[pred_idx]})
    except Exception as e:
        print("Error in prediction:", e)  # DEBUG
        return jsonify({'error': f'Prediction failed: {e}'}), 500

# ----------------- Run Server -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
