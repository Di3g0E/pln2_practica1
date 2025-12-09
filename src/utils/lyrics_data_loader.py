import os
from pathlib import Path
import pandas as pd


class LyricsDataLoader:
    def __init__(
        self,
        csv_paths: list[str] | list[Path],
        usecols: list[list[str]],
        chunk_size: int = 50000,
        max_rows: int = 300000,
        max_size_mb: float = 500
    ):
        self.csv_paths = csv_paths
        self.usecols = usecols
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        self.max_size_mb = max_size_mb

        self.__loaded_datasets = {}
        self.__load_datasets()

    def __load_datasets(self):
        for (p, usecols) in zip(self.csv_paths, self.usecols):
            csv_size_mb = os.path.getsize(p) / (1024 * 1024)
            if csv_size_mb > self.max_size_mb:
                chunks = []
                total_read = 0

                for chunk in pd.read_csv(p, usecols=usecols, chunksize=self.chunk_size):
                    chunks.append(chunk)
                    total_read += len(chunk)

                    if total_read >= self.max_rows:
                        break

                dataset = pd.concat(chunks, ignore_index=True)

                if len(dataset) > self.max_rows:
                    dataset = dataset.sample(n=self.max_rows)

            else:
                dataset = pd.read_csv(p, usecols=usecols)

            self.__loaded_datasets[str(p)] = dataset

    def get_loaded_datasets(self) -> dict[str: pd.DataFrame]:
        return self.__loaded_datasets.copy()
