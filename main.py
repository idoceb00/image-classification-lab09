import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from models.classical import train_classical_model, evaluate_classical_test
from models.cnn import train_cnn_model, evaluate_cnn_test
from utils.io import (save_metrics, save_model_weights, load_model_weights,
                      save_classical_model, load_classical_model, model_weights_exist)

def get_cifar10_loaders(batch_size=32, validation_split=0.2):
    transform = transforms.Compose([
        transforms.Resize(128), # required for Transformer models
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), # CIFAR-10 mean
                             (0.2023, 0.1994, 0.2010))  # CIFAR-10 std
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    # Validation fragment size and training fragment size
    validation_size = int(len(full_train_dataset) * validation_split)
    train_size = len(full_train_dataset) - validation_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, validation_size])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, validation_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=256)

    if model_weights_exist("classical_model.joblib"):
        clf = load_classical_model("classical_model.joblib")
    else:
        clf, acc_val = train_classical_model(train_loader, val_loader)
        save_classical_model(clf, "classical_model.joblib")

    acc_clasical_test = evaluate_classical_test(clf, test_loader)
    save_metrics("classical", acc_val, acc_clasical_test)
    #------------CNN MODEL----------------------

    # model_cnn, acc_cnn = train_cnn_model(train_loader, val_loader, num_epochs=10)
    #
    # acc_cnn_test = evaluate_cnn_test(model_cnn, test_loader)
    #
    # save_metrics("CNN(MobileNetV2)", acc_cnn, acc_cnn_test)


    weights_filename = "cnn_mobilenetv2_weights.pth"

    # ✅ Entrenar o cargar modelo según existan pesos
    if model_weights_exist(weights_filename):
        print("📦 Modelo CNN ya entrenado, cargando pesos...")

        # Crear arquitectura base
        model_cnn = models.mobilenet_v2(weights=None)
        model_cnn.classifier[1] = torch.nn.Linear(model_cnn.last_channel, 10)

        # Cargar pesos en el dispositivo correcto
        model_cnn = load_model_weights(model_cnn, weights_filename)
        acc_val = None  # (opcional) podrías cargarlo desde JSON si quisieras
    else:
        print("🚀 Entrenando modelo CNN (MobileNetV2)...")
        model_cnn, acc_val = train_cnn_model(train_loader, val_loader, num_epochs=10)

        # Guardar pesos entrenados
        save_model_weights(model_cnn, weights_filename)

    acc_test = evaluate_cnn_test(model_cnn, test_loader)


    save_metrics("cnn_mobilenetv2", acc_val, acc_test)
