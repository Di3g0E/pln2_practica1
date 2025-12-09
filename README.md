# Práctica 1 - Procesamiento de Lenguaje Natural 2
## Instrucciones de ejecución
- Para poder ejecutar este proyecto, necesitaremos Python 3.10 con una serie de dependencias que se encuentran en `pyproject.toml`.
- Para instalar las dependencias con **pip**, ejecuta el siguiente comando:
```bash
pip install -r requirements.txt
```


- Para instalar las dependencias con **uv**, ejecutamos el siguiente comando, que también creará un entorno virtual `.venv` en la raíz del proyecto con todas las dependencias instaladas:
```bash
uv sync
```

- Una vez instaladas las dependencias, podemos ejecutar el cuaderno de Jupyter `notebook.ipynb` para llevar a cabo el procesamiento de datos, entrenamiento y evaluación de los modelos y análisis de explicabilidad.

- También podemos ejecutar los scripts `train.py` y `eval.py` para entrenar, guardar y evaluar los modelos.

## Configuración de los *datasets*
- En la carpeta `config/` se encuentran los archivos de configuración para los diferentes *datasets* utilizados en la práctica.

- Estos se tratan de ficheros JSON que contienen:
    - `name`: Nombre del *dataset*.
    - `csv_name`: Nombre del fichero CSV que contiene los datos.
    - `kagglehub_handle`: Usuario y *handle* de KaggleHub donde se encuentra el *dataset*.
    - `csv_path`: Ruta dentro del ZIP del fichero CSV.
    - `cols_map`: Mapeo de las columnas del CSV a los nombres utilizados en el proyecto.
    - `language_col`: Nombre de la columna que contiene el idioma de las letras.
    - `target_language`: Idioma objetivo para filtrar las letras.
    - `vectorizer_config`: Configuración para el vectorizador, como el idioma de las stop words.
    
```json
{
    "name": "NAME",
    "csv_name": "NAME.csv",
    "kagglehub_handle": "USER/HANDLE",
    "csv_path": "CSV_PATH",
    "cols_map": {
        "lyrics": "text",
        "tag": "label",
        "language": "language"
    },
    "language_col": "language",
    "target_language": "LANGUAGE",
    "vectorizer_config": {
        "stop_words": "LANGUAGE"
    }
}
```
