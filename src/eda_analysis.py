import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Configurações visuais
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def perform_eda():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "../data/trusted/train_data.parquet")
    reports_path = os.path.join(base_path, "../reports/figures/")
    
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    print("--- INICIANDO EDA COM FOCO EM TREINO ---")
    
    try:
        df = pd.read_parquet(train_path)
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        return

    # Garante que a classe seja Inteiro (0/1) e não String ('0'/'1')
    df['class'] = df['class'].astype(int)

    # 1. Análise Temporal (Ciclos Diários)
    df['hour'] = df['time'].apply(lambda x: np.floor(x / 3600)) % 24

    plt.figure(figsize=(14, 6))
    sns.kdeplot(data=df[df['class'] == 0]['hour'], label='Legítima', fill=True, color='g', alpha=0.3)
    sns.kdeplot(data=df[df['class'] == 1]['hour'], label='Fraude', fill=True, color='r', alpha=0.3)
    plt.title('Distribuição Temporal: Fraude vs Legítima')
    plt.xlabel('Hora do Dia (Aproximada)')
    plt.legend()
    plt.savefig(os.path.join(reports_path, "1_temporal_distribution.png"))
    plt.close()
    print("✅ Gráfico Temporal gerado.")

    # 2. Análise Monetária (Amount)
    plt.figure(figsize=(10, 6))
    # Correção de Sintaxe Seaborn (hue=class)
    sns.boxplot(x='class', y='amount', data=df, hue='class', showfliers=False, palette={0: "g", 1: "r"}, legend=False) 
    plt.title('Distribuição de Valor (Log Scale Visual)')
    plt.yscale('log')
    plt.savefig(os.path.join(reports_path, "2_amount_distribution.png"))
    plt.close()
    print("✅ Gráfico de Valor gerado.")

    # 3. Ranking de Separação de Variáveis
    cols = [c for c in df.columns if c not in ['class', 'time', 'hour']]
    ranking = []
    
    for col in cols:
        mean_fraud = df[df['class'] == 1][col].mean()
        mean_legit = df[df['class'] == 0][col].mean()
        std_legit = df[df['class'] == 0][col].std()
        
        # Distância normalizada
        divergence = abs(mean_fraud - mean_legit) / (std_legit + 1e-6)
        ranking.append({'feature': col, 'divergence': divergence})
    
    ranking_df = pd.DataFrame(ranking).sort_values(by='divergence', ascending=False)
    
    print("\n--- TOP 5 FEATURES DISCRIMINANTES (Candidatas a Ouro) ---")
    print(ranking_df.head(5))
    
    # 4. Plotar as Top 3 features discriminantes
    top_features = ranking_df.head(3)['feature'].values
    
    plt.figure(figsize=(16, 5))
    for i, col in enumerate(top_features):
        plt.subplot(1, 3, i+1)
        
        sns.boxplot(x='class', y=col, data=df, hue='class', showfliers=False, palette={0: "g", 1: "r"}, legend=False)
        plt.title(f'Separação por {col}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_path, "3_top_features_separation.png"))
    plt.close()
    print("✅ Gráfico de Features Discriminantes gerado.")
    print(f"Relatórios salvos em: {reports_path}")

if __name__ == "__main__":
    perform_eda()