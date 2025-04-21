import json
import torch
from pathlib import Path
from utils.device import get_device

def save_metrics(model_name, val_acc, test_acc):
    """
    Stores metrics in a JSON file inside /results

    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    metrics = {"model": model_name,
               "val_accuracy": val_acc,
               "test_accuracy": test_acc
               }

    with open(results_dir / f"{model_name}_metrics.json", "w") as f:
       json.dump(metrics, f, indent=4)

    print(f"📊 Metricas guardadas en {results_dir / f'{model_name}_metrics.json'}")

def sava_model_weights(model, filename: str):
    """
    Saves the weights of a Pytorch model

    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    path = results_dir /filename
    torch.save(model.state_dict(), path)
    print(f"💿 Model weights stored in {path}")

def load_model_weights(model, filename: str, device=None):
    """
    Load the weights stored in a Pytorch model.

    :param model: instance of the model (must already be created with the same architecture)
    :param filename: name of the .pth file inside results/
    :param device: device where to load the weights (cpu, cuda, mps)
    """
    if device is None:
        device = get_device()

    weights_path = Path("results") / filename

    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"🔋Weights loaded from: {weights_path}")
    return model