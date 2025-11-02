import kagglehub


def download_datasets(dir_path="data", df1=True, df2=True):
    
    if df1:
        # Descargar el primer dataset (~26Mb)
        path1 = kagglehub.dataset_download("saurabhshahane/music-dataset-1950-to-2019")
        print("Dataset 1 guardado en:", path1)

    if df2:
        # Descargar el segundo dataset (~8,5Gb)
        path2 = kagglehub.dataset_download("carlosgdcj/genius-song-lyrics-with-language-information")
        print("Dataset 2 guardado en:", path2)

if __name__ == '__main__':
    download_datasets(df2=False)
