# train_cnn.py
# GPU-Accelerated CNN Training
# using ResNet-18 for ECG Classification

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import json
import time
import os
from tqdm import tqdm

# ─────────────────────────────────────
# STEP 1: GPU Setup
# ─────────────────────────────────────
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"{'='*50}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(
        0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem:.1f} GB")
print(f"{'='*50}\n")

# ─────────────────────────────────────
# STEP 2: Data Transforms
# ─────────────────────────────────────
# Training: augmentation for better generalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Testing: no augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────
# STEP 3: Load Dataset
# ─────────────────────────────────────
print("Loading datasets...")
train_dataset = datasets.ImageFolder(
    root='dataset/train',
    transform=train_transform
)
test_dataset = datasets.ImageFolder(
    root='dataset/test',
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Save class names for Flask app
class_names = train_dataset.classes
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print(f"Class names saved: {class_names}\n")

# ─────────────────────────────────────
# STEP 4: Load Pretrained ResNet-18
# ─────────────────────────────────────
print("Loading ResNet-18 pretrained model...")
model = models.resnet18(pretrained=True)

# Freeze early layers (transfer learning)
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Replace final layer for our 5 classes
num_classes = len(class_names)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# Move model to GPU
model = model.to(device)
print(f"Model loaded on {device}")

# Count trainable parameters
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
print(f"Trainable parameters: {trainable:,}\n")

# ─────────────────────────────────────
# STEP 5: Training Setup
# ─────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad,
           model.parameters()),
    lr=0.001,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

# ─────────────────────────────────────
# STEP 6: Training Loop
# ─────────────────────────────────────
EPOCHS = 15
best_accuracy = 0.0
train_losses = []
train_accuracies = []
test_accuracies = []

print(f"Starting training for {EPOCHS} epochs...")
print(f"{'='*50}")
start_time = time.time()

for epoch in range(EPOCHS):
    # ── Training Phase ──
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ── Evaluation Phase ──
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(
                labels).sum().item()

    test_acc = 100. * correct / total
    test_accuracies.append(test_acc)

    # Save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(),
                   'best_ecg_model.pth')
        print(f"  ✅ New best model saved!")

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

    scheduler.step()

# ─────────────────────────────────────
# STEP 7: Save Training Statistics
# ─────────────────────────────────────
total_time = time.time() - start_time
stats = {
    'best_accuracy': float(best_accuracy),
    'final_train_accuracy': float(
        train_accuracies[-1]),
    'training_time_seconds': round(total_time, 2),
    'device': str(device),
    'epochs': EPOCHS,
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'num_classes': num_classes,
    'class_names': class_names
}

with open('training_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Best Accuracy: {best_accuracy:.2f}%")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Device Used: {device}")
print(f"{'='*50}")
print(f"\n✅ best_ecg_model.pth saved!")
print(f"✅ training_stats.json saved!")
print(f"✅ class_names.json saved!")