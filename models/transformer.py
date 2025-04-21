import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import accuracy_score
from utils.device import get_device

def train_transformer_model(train_loader, val_loader, num_epochs=10):
    """
    Train a Transformer model (DeiT Tiny) on CIFAR-10

    :param train_loader:
    :param val_loader:
    :param num_epochs:
    :return:
    """
    device = get_device()

    model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

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

        avg_loss = running_loss / len(train_loader)
        print(f"📚Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

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
    print(f"🎯 Accuracy in validation with DeiT Tiny: {acc:.4f}")

    return model, acc