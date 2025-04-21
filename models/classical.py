import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def flatten_loader(dataloader):
    """
    Converts a Pytorch dataloader to flat Numpy arrays

    """

    x, y = [], []
    for inputs, labels in dataloader:
        x.append(inputs.view(inputs.size(0), -1).numpy()) # Flatten the images
        y.append(labels.numpy())

    return np.vstack(x), np.hstack(y)

def train_classical_model(train_loader, val_loader):
    print(f" 🏋️‍♀️Init Classical Model Training")
    x_train, y_train = flatten_loader(train_loader)
    x_val, y_val = flatten_loader(val_loader)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"🔎 Accuracy of the classical model (validation): {acc:.4f}")

    return clf, acc

def evaluate_on_test(clf, test_laoder):
    print(f"🧐 Testing the classic model")

    x_test, y_test = flatten_loader(test_laoder)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"🎯 Accuracy of the classical model in test: {acc:.4f}")

    return acc
