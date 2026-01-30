import pandas as pd
import os

def ingest_to_silver():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.normpath(os.path.join(current_dir, "../../data/raw/creditcard.csv"))
    silver_path = os.path.normpath(os.path.join(current_dir, "../../data/silver/"))

    if not os.path.exists(silver_path):
        os.makedirs(silver_path)

    try:
        # Le os dados brutos
        df = pd.read_csv(raw_path)
        
        # Padronização de Schema
        # Converter nomes de colunas para snake_case
        df.columns = [col.lower() for col in df.columns]
        
        # Persistência Idempotente 
        output_file = os.path.join(silver_path, "fraud_data_silver.parquet")
        
        # Parquet: Formato profissional para Data Science (compacto e rápido)
        df.to_parquet(output_file, index=False)
        
        print(f"✅ Ingestão concluída! Dado salvo em formato Parquet na pasta Silver.")
        
    except Exception as e:
        print(f"❌ Erro na ingestão: {e}")

if __name__ == "__main__":
    ingest_to_silver()