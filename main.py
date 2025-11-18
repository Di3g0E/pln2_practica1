from src.utils.io import download_datasets, load_data

# Descarga de los datasets si no están. Poner valores en True para descargarlos
download_datasets(dir_path="data", df1=False, df2=False)

df = load_data()

print(df.shape)
