import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_flowers_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset_path = 'data/flowers'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    api.dataset_download_files('alxmamaev/flowers-recognition', path=dataset_path, unzip=True)

    with zipfile.ZipFile(os.path.join(dataset_path, 'flowers-recognition.zip'), 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print("Dataset downloaded and extracted.")

if __name__ == "__main__":
    download_flowers_dataset()
