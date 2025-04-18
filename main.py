from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.classical import train_classical_model
from models.classical import evaluate_on_test

def get_cifar10_loaders(batch_size=64, validation_split=0.2):
    transform = transforms.Compose([
        transforms.Resize(224), # required for Transformer models
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
    train_laoder, val_loader, test_loader = get_cifar10_loaders(batch_size=256)

    clf, acc_classical = train_classical_model(train_laoder, val_loader)

    acc_test = evaluate_on_test(clf, test_loader)