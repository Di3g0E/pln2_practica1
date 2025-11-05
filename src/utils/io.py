import kagglehub
import pathlib
import os
import shutil # Necesitamos shutil para mover archivos y borrar directorios
import pandas as pd


def download_datasets(dir_path="data", df1=True, df2=True):
    
    # 1. Convertir a Path y obtener la ruta absoluta
    data_path = pathlib.Path(dir_path).resolve()
    
    # 2. Asegurarse de que el directorio exista
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Establecer la variable de entorno KAGGLEHUB_CACHE
    os.environ["KAGGLEHUB_CACHE"] = str(data_path)
    
    print(f"Directorio de destino deseado para CSVs: {data_path}")

    if df1:
        dataset_handle_1 = "saurabhshahane/music-dataset-1950-to-2019"
        print(f"\nDescargando {dataset_handle_1}...")
        
        try:
            downloaded_path_1 = kagglehub.dataset_download(dataset_handle_1)
            print(f"Descarga inicial completada en: {downloaded_path_1}")

            csv_files_1 = list(pathlib.Path(downloaded_path_1).rglob('*.csv'))
            
            if csv_files_1:
                source_csv_path_1 = csv_files_1[0]
                destination_csv_path_1 = data_path / source_csv_path_1.name
                
                print(f"Moviendo '{source_csv_path_1.name}' a '{destination_csv_path_1}'...")
                shutil.move(str(source_csv_path_1), str(destination_csv_path_1))
                print(f"CSV de Dataset 1 guardado en: {destination_csv_path_1}")
            
            else:
                print(f"No se encontró ningún archivo CSV en {downloaded_path_1} para {dataset_handle_1}.")
        
        except Exception as e:
            print(f"Error al descargar o procesar {dataset_handle_1}: {e}")

    if df2:
        dataset_handle_2 = "carlosgdcj/genius-song-lyrics-with-language-information"
        print(f"\nDescargando {dataset_handle_2}...")

        try:
            downloaded_path_2 = kagglehub.dataset_download(dataset_handle_2)
            print(f"Descarga inicial completada en: {downloaded_path_2}")

            csv_files_2 = list(pathlib.Path(downloaded_path_2).rglob('*.csv'))
            
            if csv_files_2:
                source_csv_path_2 = csv_files_2[0] # Asumiendo un solo CSV
                destination_csv_path_2 = data_path / source_csv_path_2.name
                
                print(f"Moviendo '{source_csv_path_2.name}' a '{destination_csv_path_2}'...")
                shutil.move(str(source_csv_path_2), str(destination_csv_path_2))
                print(f"CSV de Dataset 2 guardado en: {destination_csv_path_2}")

            else:
                print(f"No se encontró ningún archivo CSV en {downloaded_path_2} para {dataset_handle_2}.")
        
        except Exception as e:
            print(f"Error al descargar o procesar {dataset_handle_2}: {e}")

    # --- SECCIÓN DE LIMPIEZA FINAL ---
    # Una vez movidos todos los CSVs, borramos el directorio 'datasets'
    # que 'kagglehub' creó dentro de nuestro 'data_path'.
    
    cleanup_path = data_path / "datasets"
    
    if cleanup_path.exists() and cleanup_path.is_dir():
        print(f"\nLimpiando directorio de caché de kagglehub: {cleanup_path}")
        try:
            # shutil.rmtree borra un directorio y todo su contenido
            shutil.rmtree(cleanup_path)
            print("Limpieza completada.")
        except OSError as e:
            print(f"Error al limpiar el directorio {cleanup_path}: {e}")

def cargar_dataset(filename="tcc_ceds_music.csv"):
    """
    Carga un dataset (archivo .csv) desde la carpeta 'data'.

    Args:
        filename (str): El nombre del archivo (ej. "archive.csv").

    Returns:
        pd.DataFrame: El DataFrame cargado, o None si hay un error.
    """
    
    # --- 1. Calcular la ruta a la carpeta 'data' ---
    # Esta lógica es la misma que usamos en el bloque __main__
    
    try:
        # Ruta de este script (io.py)
        script_file_path = pathlib.Path(__file__)
        # Ruta de la raíz del proyecto ('..')
        project_root = script_file_path.parent.parent.parent
        # Ruta del directorio 'data'
        data_dir = project_root + "/" + "data"
    except NameError:
        # __file__ no está definido si se ejecuta en un notebook/REPL
        # Asumimos una ruta relativa estándar
        print("Advertencia: __file__ no está definido. Usando ruta relativa './data'")
        data_dir = pathlib.Path("./data")
        
    # --- 2. Construir la ruta completa al archivo ---
    file_path = data_dir + "/" + filename
    
    # --- 3. Comprobar si el archivo existe y cargarlo ---
    if not file_path.exists():
        print(f"Error: El archivo no se encontró en: {file_path}")
        print(f"Asegúrate de que '{filename}' exista en la carpeta 'data'.")
        return None
        
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset '{filename}' cargado exitosamente.")
        return df
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        return None



# if __name__ == '__main__':
#     # Calcular la ruta de la carpeta 'data' en la raíz del proyecto
#     script_file_path = pathlib.Path(__file__)
#     project_root = script_file_path.parent.parent.parent
#     data_dir_path = project_root / "data"

#     # Llamamos solo con df1=True (df2=False por defecto)
#     download_datasets(dir_path=data_dir_path, df1=True, df2=False)
