import os
from pathlib import Path
import shutil
import kagglehub


class LyricsDatasetManager:
    """
    Clase dedicada a la obtención de los *datasets* utilizados
    """

    def __init__(
            self,
            directory: str | Path = "data"
    ):
        """
        Constructor de la clase DatasetManager.
        Args:
            dir_path (str | Path): directorio en que se almacenarán el/los *dataset/s*. Por defecto,
                                   se utiliza `./data`
        """
        self.directory = Path(directory).resolve()
        self.__datasets_paths = []
        self.__prepare_directory()

    def __prepare_directory(self):
        """
        Crea el directorio en que se almacenarán los datasets y establece
        el directorio de la caché de `kagglehub`.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(self.directory)
        print(
            f"- Directorio que contiene los datasets en CSV: {self.directory}")

    def __download_and_move_csv(
            self,
            dataset_handle: str,
            dataset_label: str
    ) -> Path:
        print(f"- Descargando '{dataset_handle}'...")

        try:
            downloaded_path = kagglehub.dataset_download(dataset_handle)

            csv_files = list(Path(downloaded_path).rglob('*.csv'))

            if csv_files:
                if len(csv_files) > 1:
                    for source_csv in csv_files:
                        if "train" in source_csv.name:
                            destination_csv = self.directory / \
                                (dataset_label + ".csv")

                            print(
                                f"- Moviendo '{source_csv.name}' a '{destination_csv}'...")
                            shutil.move(str(source_csv), str(destination_csv))
                            return destination_csv

                else:
                    source_csv = csv_files[0]
                    destination_csv = self.directory / (dataset_label + ".csv")
                    print(
                        f"- Moviendo '{source_csv.name}' a '{destination_csv}'...")
                    shutil.move(str(source_csv), str(destination_csv))
                    return destination_csv

            else:
                print(
                    f"- No se encontró ningún archivo CSV en {downloaded_path} para {dataset_handle}.")

        except Exception as e:
            print(f"- Error al descargar o procesar {dataset_handle}: {e}")

    def __cleanup_kagglehub_cache(self):
        """
        Elimina la caché de kagglehub.
        """
        cleanup_path = self.directory / "datasets"
        if cleanup_path.exists() and cleanup_path.is_dir():
            print(
                f"- Limpiando directorio de caché de kagglehub: {cleanup_path}")
            try:
                shutil.rmtree(cleanup_path)
                print("- Limpieza completada.")
            except OSError as e:
                print(f"- Error al limpiar el directorio {cleanup_path}: {e}")

    def download_lyrics_datasets(
            self,
            df1: bool = True,
            df2: bool = False
    ) -> list[Path]:
        """
        Descarga los *datasets* utilizados en el proyecto.
        Args:
            df1 (bool): determina si se descarga el primer *dataset* (Genius Song Lyrics). Por defecto es `True`.
            df2 (bool): determina si se descarga el segundo *dataset* (Multi-Lingual Lyrics for Genre Classification).
                        Por defecto es `True`.
        """
        if df1:
            destination_paths = self.__download_and_move_csv(
                dataset_handle="carlosgdcj/genius-song-lyrics-with-language-information",
                dataset_label="genius-lyrics"
            )
            self.__datasets_paths.append(destination_paths)

        if df2:
            destination_paths = self.__download_and_move_csv(
                dataset_handle="mateibejan/multilingual-lyrics-for-genre-classification",
                dataset_label="multi_lingual-lyrics"
            )
            self.__datasets_paths.append(destination_paths)

        self.__cleanup_kagglehub_cache()

        return self.get_datasets_paths()

    def get_datasets_paths(self) -> list[list[Path]]:
        """
        Devuelve las rutas a los *datasets* en CSV.
        Output:
            list[list[Path]]
        """
        return self.__datasets_paths.copy()
