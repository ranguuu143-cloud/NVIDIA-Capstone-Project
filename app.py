# app.py — Flask Web Application
# ECG Cardiac Abnormality Detection

from flask import (Flask, render_template,
                   request, jsonify)
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import os
import time

app = Flask(__name__)

# ─────────────────────────────────────
# Configuration
# ─────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────
# Load Model
# ─────────────────────────────────────
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# Load class names
with open('class_names.json') as f:
    CLASS_NAMES = json.load(f)

# Load training stats
with open('training_stats.json') as f:
    STATS = json.load(f)

# Class descriptions for results page
CLASS_INFO = {
    'Fusion': {
        'risk': 'MODERATE RISK',
        'color': 'orange',
        'description': 'Fusion beat detected. '
            'Combination of normal and '
            'abnormal heartbeat patterns.',
        'advice': 'Schedule a cardiology '
            'appointment for further evaluation.'
    },
    'Normal': {
        'risk': 'LOW RISK',
        'color': 'green',
        'description': 'Normal sinus rhythm '
            'detected. Heart appears to be '
            'functioning normally.',
        'advice': 'Continue regular health '
            'checkups and maintain healthy lifestyle.'
    },
    'Supraventricular': {
        'risk': 'MODERATE RISK',
        'color': 'orange',
        'description': 'Supraventricular '
            'abnormality detected. Irregular '
            'electrical activity above ventricles.',
        'advice': 'Consult a cardiologist. '
            'Further ECG monitoring recommended.'
    },
    'Unknown': {
        'risk': 'NEEDS REVIEW',
        'color': 'blue',
        'description': 'Unclassified pattern '
            'detected. Signal requires manual '
            'expert review.',
        'advice': 'Please consult a cardiologist '
            'for manual ECG interpretation.'
    },
    'Ventricular': {
        'risk': 'HIGH RISK',
        'color': 'red',
        'description': 'Ventricular abnormality '
            'detected. Potentially serious '
            'irregular heartbeat pattern.',
        'advice': 'Seek immediate medical '
            'attention from a cardiologist.'
    }
}

# Build ResNet-18 model
def load_model():
    m = models.resnet18(pretrained=False)
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES))
    )
    m.load_state_dict(
        torch.load('best_ecg_model.pth',
                   map_location=device)
    )
    m.to(device)
    m.eval()
    return m

model = load_model()
print(f"Model loaded on {device}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower()
            in ALLOWED_EXTENSIONS)

def predict_ecg(image_path):
    """Run prediction on uploaded ECG image"""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    inference_time = round(
        (time.time() - start) * 1000, 2
    )

    class_name = CLASS_NAMES[predicted.item()]
    conf_pct = round(confidence.item() * 100, 2)
    all_probs = {
        CLASS_NAMES[i]: round(
            probabilities[0][i].item() * 100, 2
        )
        for i in range(len(CLASS_NAMES))
    }

    return {
        'class': class_name,
        'confidence': conf_pct,
        'all_probabilities': all_probs,
        'inference_time_ms': inference_time,
        'device': str(device)
    }

# ─────────────────────────────────────
# Routes
# ─────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html',
                           stats=STATS)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/model_info')
def model_info():
    return render_template('model_info.html',
                           stats=STATS)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'ecg_image' not in request.files:
        return render_template('predict.html',
            error='No file uploaded!')

    file = request.files['ecg_image']

    if file.filename == '':
        return render_template('predict.html',
            error='No file selected!')

    if not allowed_file(file.filename):
        return render_template('predict.html',
            error='Please upload PNG or JPG only!')

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'], filename
    )
    file.save(filepath)

    # Run prediction
    result = predict_ecg(filepath)
    class_info = CLASS_INFO.get(
        result['class'],
        CLASS_INFO['Unknown']
    )

    return render_template(
        'result.html',
        result=result,
        class_info=class_info,
        image_path=filename
    )

if __name__ == '__main__':
    app.run(debug=False)