import kagglehub
import os
import shutil
import pandas as pd

def fetch_fraud_data():
    """
    Realiza a ingestão dos dados de fraude via API moderna.
    Simula a extração de um Data Lake par o ambiente local.
    """
    print("Iniciando a extração dos dados...")

    # Faz o download da última versão do dataset
    # Caminho retornado é onde o kagglehub salva temporariamente
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    # Define a pasta 'landing zone' (dados brutos)
    target_folder = ".data/raw"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Move o arquivo para a estrutura de projeto para manter o versionamento
    filename = "creditcard.csv"
    src_file = os.path.join(path, filename)
    dest_file = os.path.join(target_folder, filename)

    if os.path.exists(src_file):
        shutil.copy(src_file, dest_file)
        print(f" Sucesso! Dados salvos em: {dest_file}")
    else:
        print("Erro: Arquivo não encontrado.")

if __name__=="__main__":
    fetch_fraud_data()