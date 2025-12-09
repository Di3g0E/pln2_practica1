import random
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib

from src.utils.lyrics_data_processor import LyricsDataProcessor
from src.models.baseline_model import BaselineModel
from src.models.transformer_model import TransformerModel


def main():
    print("- Evaluación de los modelos entrenados")

    # Mostrar versiones de las librerías
    print("- Versiones de las librerías utilizadas:")
    print(f"    - NumPy: {np.__version__}")
    print(f"    - Pandas: {pd.__version__}")
    print(f"    - scikit-learn: {sklearn.__version__}")
    print(f"    - PyTorch: {torch.__version__}")
    print(f"    - Matplotlib: {matplotlib.__version__}")

    # Dispositivo para PyTorch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Semilla
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    sklearn.random.seed(SEED)

    # Directorios
    PREPROCESSED_DATA_DIR = Path("data/generated")
    MODELS_DIR = Path("models")

    # Mapeo de géneros (labels)
    LABEL2ID = {
        "rap": 0,
        "rock": 1,
        "pop": 2,
        "other": 3
    }
    ID2LABEL = {
        0: "rap",
        1: "rock",
        2: "pop",
        3: "other"
    }

    # Cargar datasets preprocesados
    X_test_baseline = LyricsDataProcessor.load_data_baseline(
        PREPROCESSED_DATA_DIR / "X_test_baseline.npz")
    y_test_baseline = LyricsDataProcessor.load_label(
        PREPROCESSED_DATA_DIR / "y_test.npy")
    test_dataset_transformer = LyricsDataProcessor.load_dataset_transformer(
        PREPROCESSED_DATA_DIR / "test_dataset_transformer")

    # Cargar y evaluar el modelo baseline
    baseline_model = BaselineModel.load(MODELS_DIR / "baseline_model.pkl")

    # Evaluamos el modelo Baseline
    baseline_results = baseline_model.evaluate(
        X_test_baseline, y_test_baseline)
    print("- Resultados del modelo Baseline:")
    print(f"    - Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"    - F1 Macro: {baseline_results['f1_macro']:.4f}")
    for idx, f1 in enumerate(baseline_results['f1_per_class']):
        print(f"        - F1 Clase {ID2LABEL[idx]}: {f1:.4f}")

    # Cargar y evaluar el modelo transformer
    transformer_model = TransformerModel.load_model(
        MODELS_DIR / "transformer_model")

    # Evaluamos el modelo Transformer
    transformer_results = transformer_model.evaluate(test_dataset_transformer)
    print("- Resultados del modelo Transformer:")
    print(f"    - Accuracy: {transformer_results['accuracy']:.4f}")
    print(f"    - F1 Macro: {transformer_results['f1_macro']:.4f}")
    for idx, f1 in enumerate(transformer_results['f1_per_class']):
        print(f"        - F1 Clase {ID2LABEL[idx]}: {f1:.4f}")


if __name__ == "__main__":
    main()
