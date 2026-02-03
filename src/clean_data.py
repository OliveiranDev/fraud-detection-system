import pandas as pd
import os
import numpy as np

def clean_and_split_data():
    """
    Realiza a limpeza, deduplicação e split temporal dos dados.
    Entrada: Silver (Parquet)
    Saída: Trusted (Train/Test Parquet separados)
    """
    # Caminhos
    base_path = os.path.dirname(os.path.abspath(__file__))
    silver_path = os.path.join(base_path, "../data/silver/fraud_data_silver.parquet")
    trusted_path = os.path.join(base_path, "../data/trusted/")
    
    if not os.path.exists(trusted_path):
        os.makedirs(trusted_path)
        
    print("--- INICIANDO LIMPEZA E TRATAMENTO (ETAPA 4) ---")
    
    # 1. Carregamento
    try:
        df = pd.read_parquet(silver_path)
        print(f"Dados carregados: {df.shape[0]} registros.")
    except Exception as e:
        print(f"Erro ao carregar dados Silver: {e}")
        return

    # 2. Deduplicação (Crítico para Fraude)
    # Transações idênticas (mesmo tempo, valor e features V) são prováveis erros de sistema/retry
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    cleaned_rows = df.shape[0]
    duplicates_removed = initial_rows - cleaned_rows
    
    print(f"Duplicatas removidas: {duplicates_removed}")
    if duplicates_removed > 0:
        print("⚠️ ALERTA: Duplicatas encontradas.")

    # 3. Tratamento de Valores Ausentes
    # Regra: Se houver NAs, imputar com -1 ou média (mas a média DEVE ser calculada apenas no treino depois).
    if df.isnull().sum().max() > 0:
        print("Valores nulos detectados. Tratando...")
        df.fillna(0, inplace=True)
    
    # 4. Estratégia de Split (Time-Based Split)
    df = df.sort_values(by="time")
    
    # Definindo ponto de corte (80% para treino, 20% para teste - cronológico)
    split_index = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print("\n--- SPLIT TEMPORAL ---")
    print(f"Treino (Passado): {train_df.shape[0]} transações | Fraudes: {train_df['class'].sum()}")
    print(f"Teste (Futuro):   {test_df.shape[0]} transações  | Fraudes: {test_df['class'].sum()}")
    
    # Validação de Sanidade do Split
    if test_df['time'].min() < train_df['time'].max():
        raise ValueError("ERRO CRÍTICO: Vazamento de tempo detectado. O teste contém dados do passado.")

    # 5. Persistência na Camada Trusted
    train_df.to_parquet(os.path.join(trusted_path, "train_data.parquet"), index=False)
    test_df.to_parquet(os.path.join(trusted_path, "test_data.parquet"), index=False)
    
    print(f"\n✅ Dados limpos e divididos salvos em: {trusted_path}")
    print("Pronto para EDA e Feature Engineering.")

if __name__ == "__main__":
    clean_and_split_data()