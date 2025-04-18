# 🧪 Image Classification Lab 09 – CIFAR-10

Este proyecto corresponde al **Laboratorio 09** de la asignatura de Visión Artificial en la Universidad de León. El objetivo es entrenar y evaluar distintos modelos de clasificación de imágenes sobre el dataset **CIFAR-10**, incluyendo:

- Un modelo clásico de ML
- Un modelo basado en CNNs
- Un modelo basado en Transformers

---

## 📂 Estructura del proyecto
## 📁 Estructura del Proyecto

- `data_loader.py`: Carga y partición del dataset CIFAR-10 (train/val/test)
- `main.py`: Script principal que entrena y evalúa los 3 modelos
- `models/`: Modelos organizados por tipo
  - `classical.py`: Modelo clásico (SVM, Random Forest, MLP...)
  - `cnn.py`: Modelo CNN (ResNet, VGG, etc.)
  - `transformer.py`: Modelo Transformer (DeiT, Swin, MobileViT...)
- `utils/`: Funciones auxiliares
  - `train_eval.py`: Entrenamiento, evaluación, métricas, etc.
- `results/`: Resultados del entrenamiento
  - `plots/`: Curvas de loss y accuracy
  - `checkpoints/`: Pesos de modelos entrenados (opcional)
  - `metrics.json`: Accuracy y métricas finales
- `report.pdf`: Informe con descripción de métodos y métricas
- `environment.yml`: Dependencias del entorno Conda
- `README.md`: Este archivo
---

## 📚 Dataset

Se utiliza el dataset **CIFAR-10**, que contiene 60,000 imágenes a color (32x32) en 10 clases. Se divide de la siguiente forma:

- **Train**: 80% del conjunto de entrenamiento original
- **Validation**: 20% del conjunto de entrenamiento original
- **Test**: conjunto oficial de test de 10,000 imágenes

---

## 🧠 Modelos implementados

| Modelo       | Herramienta       | Objetivo mínimo |
|--------------|-------------------|------------------|
| Clásico      | `scikit-learn`    | ≥ 45% Accuracy   |
| CNN          | `torchvision`     | ≥ 85% Accuracy   |
| Transformer  | `timm`            | ≥ 80% Accuracy   |

---

## 🚀 Requisitos

- Python 3.10+
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib

### Instalación recomendada (Conda)

```bash
conda create -n lab09 python=3.10
conda activate lab09
conda install pytorch torchvision torchaudio -c pytorch
pip install timm scikit-learn matplotlib