import random
import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib

from pathlib import Path

from src.utils.lyrics_dataset_manager import LyricsDatasetManager
from src.utils.lyrics_dataset_config import LyricsDatasetConfig
from src.utils.lyrics_data_loader import LyricsDataLoader
from src.utils.lyrics_data_processor import LyricsDataProcessor
from src.models.baseline_model import BaselineModel
from src.models.transformer_model import TransformerModel


def main():
    print("- Entrenamiento de los modelos")

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

    # ¿Descargar datasets?
    DOWNLOAD_DATASETS = False

    # Directorios
    DATA_DIR = Path("data")
    TRANSFORMER_OUTPUT_DIR = Path("transformer_output")

    # ¿Datasets preprocesados?
    USE_PREPROCESSED_DATASETS = True
    PREPROCESSED_DATA_DIR = Path(DATA_DIR) / "generated"

    # ¿Entrenar?
    TRAIN_MODEL = False

    # Splits del dataset
    EVAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Mapeo de géneros (labels)
    GENRE_MAP = {
        # Géneros principales
        "rap": "rap",
        "hip-hop": "rap",
        "rock": "rock",
        "pop": "pop",

        # Otros géneros
        "rb": "other",
        "r&b": "other",
        "metal": "other",
        "indie": "other",
        "folk": "other",
        "jazz": "other",
        "electronic": "other",
        "edm": "other",
        "country": "other",
        "misc": "other",
        "soul": "other",
        "blues": "other",
        "classical": "other",
    }
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

    # Configuración de los datasets
    GENIUS_CONFIG_PATH = Path("config/genius-song-lyrics.json")
    MULTILINGUAL_CONFIG_PATH = Path(
        "config/multi_lingual-lyrics-for-genre-classification.json")

    # Modelo basado en Transformer
    TRANSFORMER_MODEL_NAME = "roberta-base"

    # Mostrar parámetros
    print(f"- Dispositivo utilizado por PyTorch: {DEVICE}")
    print(f"- Semilla utilizada para reproducibilidad: {SEED}")
    print(f"- Descarga de datasets: {'Sí' if DOWNLOAD_DATASETS else 'No'}")

    # Descargar los datasets si es necesario
    if DOWNLOAD_DATASETS:
        dataset_manager = LyricsDatasetManager(directory=DATA_DIR)
        dataset_manager.download_lyrics_datasets(df1=True, df2=True)
        dataset_paths = dataset_manager.get_datasets_paths()

    else:
        dataset_paths = [DATA_DIR/"genius-lyrics.csv",
                         DATA_DIR/"multi_lingual-lyrics.csv"]

    # Obtenemos las configuraciones de los datasets
    genius_config = LyricsDatasetConfig(config_path=GENIUS_CONFIG_PATH)
    multilingual_config = LyricsDatasetConfig(
        config_path=MULTILINGUAL_CONFIG_PATH)

    # Cargamos los datasets
    data_loader = LyricsDataLoader(
        csv_paths=dataset_paths,
        usecols=[genius_config.cols_map.keys(
        ), multilingual_config.cols_map.keys()]
    )
    datasets = list(data_loader.get_loaded_datasets().values())

    # Preprocesamos el dataset
    data_processor = LyricsDataProcessor(
        output_dir=TRANSFORMER_OUTPUT_DIR,
        datasets=datasets,
        dataset_configs=[genius_config, multilingual_config],
        genre_map=GENRE_MAP,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        eval_split=EVAL_SPLIT,
        test_split=TEST_SPLIT,
        transformer_model_name=TRANSFORMER_MODEL_NAME,
        device=DEVICE
    )
    data_processor.harmonize_pipeline()

    # Guardamos los datos preprocesados
    data_processor.save_all()

    # Cargamos los datos
    X_train_baseline = LyricsDataProcessor.load_data_baseline(
        DATA_OUTPUT_DIR / "X_train_baseline.npz")
    X_eval_baseline = LyricsDataProcessor.load_data_baseline(
        DATA_OUTPUT_DIR / "X_eval_baseline.npz")
    y_train = LyricsDataProcessor.load_label(DATA_OUTPUT_DIR / "y_train.npy")
    y_eval = LyricsDataProcessor.load_label(DATA_OUTPUT_DIR / "y_eval.npy")

    train_dataset_transformer = LyricsDataProcessor.load_dataset_transformer(
        DATA_OUTPUT_DIR / "train_dataset_transformer")
    eval_dataset_transformer = LyricsDataProcessor.load_dataset_transformer(
        DATA_OUTPUT_DIR / "eval_dataset_transformer")

    # Entrenamos el modelo baseline
    baseline_model = BaselineModel()
    baseline_model.fit(X_train_baseline, y_train, X_eval_baseline, y_eval)

    # Entrenamos el modelo transformer
    transformer_model = TransformerModel(
        TRANSFORMER_OUTPUT_DIR, LABEL2ID, ID2LABEL)
    transformer_model.fit(train_dataset_transformer, eval_dataset_transformer)


if __name__ == "__main__":
    main()
