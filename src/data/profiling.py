import pandas as pd
import os

def run_initial_profiling():
    # Caminho do dado bruto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, "../../data/raw/creditcard.csv")
    raw_data_path = os.path.normpath(raw_data_path)

    if not os.path.exists(raw_data_path):
        print(f"ERRO: Arquivo não encontrado em {raw_data_path}")
        print(f"CWD (Diretório de Trabalho Atual): {os.getcwd()}")
        return
    
    # Carrega apenas uma amostra para análise de saúde
    df = pd.read_csv(raw_data_path)
    
    print("---RELATÓRIO DE SAÚDE DOS DADOS (FASE 2)---")

    # Avaliação de Volume e Granularidade
    print(f"\nTotal de Transações: {df.shape[0]}")
    print(f"\nTotal de Atributos: {df.shape[1]}")

    # Avaliação de Dados Ausentes
    null_counts = df.isnull().sum().sum()
    print(f"Valores Ausentes: {null_counts} ({'Saudável' if null_counts == 0 else 'Alerta'})")

    # Avaliação de Desbalanceamento
    fraud_count = df[df['Class'] == 1].shape[0]
    legit_count = df[df['Class'] == 0].shape[0]
    fraud_percent = (fraud_count / len(df)) * 100

    print(f"\n---ANÁLISE DA VARIÁVEL ALVO---")
    print(f"Transações Legítimas: {legit_count}")
    print(f"Transações Fraudulentas: {fraud_count}")
    print(f"Taxa de Fraude: {fraud_percent:.4f}%")

    # Avaliação Temporal
    print(f"\nJanela Temporal (segundos): {df['Time']. max()} (~{df['Time'].max()/3600:.2f} horas)")

if __name__== "__main__":
    run_initial_profiling()