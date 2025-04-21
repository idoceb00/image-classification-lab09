import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
import numpy as np

def train_cnn_model(train_loader, val_loader, num_epochs=10, device="cpu"):
    model = models.mobilenet_v2(pretrained=True)

    model.classifier[1] = nn.Linear(model.last_channel, 10)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss /len(train_loader)
        print(f"📚Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"🎯 Accuracy in validation with MobileNetV2: {acc:.4f}")

    return model, acc