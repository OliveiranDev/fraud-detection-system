import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

# Cria classe para produção futura
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Feature: Hora do Dia (0-23)
        # Assume ciclos de 24h.
        X['hour'] = X['time'].apply(lambda x: np.floor(x / 3600)) % 24
        
        # 2. Feature: Is Night (Madrugada) - Insight da EDA
        # Transações legítimas caem drasticamente entre 0h e 6h.
        X['is_night'] = X['hour'].apply(lambda x: 1 if x <= 6 else 0)
        
        # 3. Transformação de Log no Amount.
        # Adicionamos +0.001 para evitar log(0)
        X['amount_log'] = np.log(X['amount'] + 0.001)
        
        return X

def run_feature_engineering():
    base_path = os.path.dirname(os.path.abspath(__file__))
    trusted_path = os.path.join(base_path, "../data/trusted/")
    gold_path = os.path.join(base_path, "../data/gold/")
    
    if not os.path.exists(gold_path):
        os.makedirs(gold_path)
        
    print("--- INICIANDO FEATURE ENGINEERING ---")
    
    # Processa Treino e Teste separadamente, usando a MESMA lógica
    files = ["train_data.parquet", "test_data.parquet"]
    
    engineer = FraudFeatureEngineer()
    
    for file in files:
        input_file = os.path.join(trusted_path, file)
        output_file = os.path.join(gold_path, file)
        
        try:
            df = pd.read_parquet(input_file)
            print(f"Processando {file} ({df.shape[0]} registros)...")
            
            # Aplica a engenharia
            df_gold = engineer.transform(df)
            
            # Salva na camada Gold
            df_gold.to_parquet(output_file, index=False)
            print(f"✅ Salvo em Gold: {output_file} | Colunas: {df_gold.columns.tolist()[-3:]}")
            
        except Exception as e:
            print(f"❌ Erro em {file}: {e}")

if __name__ == "__main__":
    run_feature_engineering()