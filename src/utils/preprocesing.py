import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from src.utils.io import DatasetManager # Importamos tu clase existente
# import pathlib

class DataProcessor:
    def __init__(self, dataset_manager=DatasetManager(), seed=42):
        self.dm = dataset_manager
        self.df = None
        self.seed = seed
        
    def load_and_unify(self, df_name="en_song_lyrics.csv"):
        """
        Target Schema: ['text', 'label']
        """

        # Asumimos que ya ejecutamos dm.download()
        try:
            df = self.dm.load(df_name) # Ajusta el nombre según la descarga real
            # Renombrar columnas al estándar
            self.df = df.rename(columns={'lyrics': 'text', 'tag': 'label'})
            print(f"Dataset cargado: {self.df.shape}")
        
        except Exception as e:
            print(f"Aviso: No se pudo cargar datos: {e}")

    def normalize_labels(self, threshold=100000):
        """
        Punto 3 revisado: Normaliza texto y convierte a valores numéricos.
        Entrada: ['rap', 'rb', 'rock', 'pop', 'misc', 'country']
        Salida: Enteros [0, 1, 2, 3, 4]
        """
        if self.df is None: return

        # 1. Limpieza básica de strings
        self.df['label'] = self.df['label'].astype(str).str.strip().str.lower()

        # 2. Definir mapa de etiquetas (Ajusta esto según tus datos reales)
        label_map = {
            'rap': 'Hip Hop',
            'rb': 'R&B',
            'rock': 'Rock',
            'pop': 'Pop',
            'country': 'Country'
            # 'misc' se ignora aquí para que se convierta en NaN y se elimine
        }
        
        print("\nEstandarizando nombres de etiquetas...")
        # Creamos una columna temporal para el nombre legible
        self.df['label_name'] = self.df['label'].map(label_map)
        
        # Eliminamos 'misc' o etiquetas no mapeadas (NaN)
        n_dropped = self.df['label_name'].isna().sum()
        if n_dropped > 0:
            print(f"Eliminando {n_dropped} filas (misc/desconocidos).")
            self.df = self.df.dropna(subset=['label_name'])

        # 2. Conversión a Numérico (Label Encoding)
        # Ordenamos alfabéticamente para asegurar determinismo: Country=0, Hip Hop=1, etc.
        unique_labels = sorted(self.df['label_name'].unique())
        
        # Creamos los diccionarios de mapeo
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for i, label in enumerate(unique_labels)}
        
        print(f"Asignando IDs numéricos: {self.label2id}")
        
        # Sobrescribimos la columna 'label' con el entero
        self.df['label'] = self.df['label_name'].map(self.label2id)
        
        # Borramos la columna temporal de nombres para ahorrar memoria
        self.df = self.df.drop(columns=['label_name']) 
        
        # Filtrar clases con muy pocos ejemplos (ruido)
        conteo = self.df['label'].value_counts()
        clases_validas = conteo[conteo > (self.df.shape[0] * threshold)].index # Umbral mínimo
        self.df = self.df[self.df['label'].isin(clases_validas)]
        
        print(f"Clases resultantes: {self.df['label'].unique()}")

    def clean_and_deduplicate(self, min_len=25):
        """
        Punto 4: Deduplicar textos para evitar data leakage.
        """
        print("\nLimpiando y deduplicando...")
        original_size = len(self.df)
        
        # Eliminar nulos
        self.df = self.df.dropna(subset=['text', 'label'])
        
        # Eliminar duplicados exactos en el texto
        self.df = self.df.drop_duplicates(subset=['text'], keep='first')
        
        # Limpieza básica de texto (eliminar etiquetas de metadatos comunes en lyrics)
        # Ejemplo: [Chorus], [Intro]
        self.df['text'] = self.df['text'].str.replace(r'\[.*?\]', '', regex=True)
        
        # Eliminar textos demasiado cortos tras limpieza
        self.df = self.df[self.df['text'].str.len() > min_len]
        
        final_size = len(self.df)
        print(f"--> Filas eliminadas: {original_size - final_size}")
        print(f"--> Dataset limpio: {final_size}")

    def handle_imbalance(self, method='sampling', max_per_class=10000):
        """
        Punto 5: Controlar el desbalance.
        Estrategia: Downsampling de la clase mayoritaria para no explotar en entrenamiento.
        """
        print("\nGestionando desbalance de clases...")
        conteo = self.df['label'].value_counts()
        print("Distribución original:")
        print(conteo)

        if method == 'sampling':
            # Downsampling estratificado
            dfs_list = []
            for label, group in self.df.groupby('label'):
                if len(group) > max_per_class:
                    dfs_list.append(group.sample(max_per_class, random_state=self.seed))
                else:
                    dfs_list.append(group)
            
            self.df = pd.concat(dfs_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)
            print("\nDistribución tras sampling:")
            print(self.df['label'].value_counts())
            
    def save(self, filename="dataset_harmonized.csv"):
        self.dm.save_dataset(self.df, filename)

    def get_df(self):
        return self.df
