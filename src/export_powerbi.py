import pandas as pd
import joblib
import os
import numpy as np

def export_data_for_business_simulation():
    print("--- 🏢 GERANDO SIMULAÇÃO DE NEGÓCIO (JAN/2026) ---")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    #Insere dados de treino e teste
    path_test = os.path.join(base_path, "../data/gold/test_data.parquet")
    path_train = os.path.join(base_path, "../data/gold/train_data.parquet")
    model_path = os.path.join(base_path, "../models/challenger_model.pkl")
    output_path = os.path.join(base_path, "../reports/powerbi_dataset.csv")
    
    try:
        print("1. Carregando histórico de transações...")
        
        # Carrega as partes
        if os.path.exists(path_train) and os.path.exists(path_test):
            df_train = pd.read_parquet(path_train)
            df_test = pd.read_parquet(path_test)
            
            df = pd.concat([df_train, df_test], axis=0).sort_values(by='time')
            print(f"   -> Dataset Unificado: {len(df)} transações (Simulação Total)")
        else:
            print("   ❌ Erro: Arquivos de dados não encontrados. Verifique a pasta data/gold.")
            return

        # Carrega o Modelo
        model = joblib.load(model_path)
        
        # 2. Converte para Hora do Dia (0-23).
        print("2. Calculando Ciclo de 24h...")
        df['hour'] = (df['time'] // 3600) % 24
        df['hour'] = df['hour'].astype(int)
        
        # Definição de Períodos de Negócio
        def get_period(h):
            if h <= 6: return 'Madrugada'
            elif h >= 18: return 'Noite'
            else: return 'Dia Comercial'
        df['Periodo'] = df['hour'].apply(get_period)

        # Log Amount
        df['amount_log'] = np.log(df['amount'] + 0.001)
        df['is_night'] = df['hour'].apply(lambda x: 1 if x <= 6 else 0)

        # 3. Simula Inferência
        print("3. Executando Modelo em todo o período...")
        expected_cols = model.feature_names_in_
        X = df[expected_cols]
        probs = model.predict_proba(X)[:, 1]
        
        # 4. Monta Relatório Final
        export_df = df[['time', 'amount', 'class', 'hour', 'Periodo']].copy()
        
        # ID Único Sequencial
        export_df['transaction_id'] = range(1, len(export_df) + 1)
        
        # Probabilidade
        export_df['probability'] = np.round(probs, 4)
        
        # Status Real
        export_df['transaction_status'] = np.where(export_df['class'] == 1, 'Fraude Real', 'Legítima')
        
        # --- CÁLCULO DE IMPACTO FINANCEIRO (SIMULAÇÃO) ---
        COST_FRAUD = 100    # Custo médio de um chargeback
        COST_BLOCK = 2      # Custo operacional de revisar/bloquear cliente
        
        # Cenário A: Sem Modelo (Deixamos tudo passar)
        # Custo = Todas as fraudes viram prejuízo
        export_df['cost_no_model'] = export_df['class'] * COST_FRAUD
        
        # Cenário B: Modelo Atual (Threshold 0.20 -Estratégia Aplicada)
        pred_20 = (probs >= 0.20).astype(int)
        costs_20 = []
        
        # Lógica de Custo linha a linha
        for real, pred in zip(export_df['class'], pred_20):
            if real == 1 and pred == 1:
                costs_20.append(0)          # Sucesso! Bloqueamos a fraude. Custo zero (ou quase zero).
            elif real == 1 and pred == 0:
                costs_20.append(COST_FRAUD) # Falha (FN). Fraude passou. Prejuízo total.
            elif real == 0 and pred == 1:
                costs_20.append(COST_BLOCK) # Atrito (FP). Bloqueamos cliente bom. Custo baixo.
            else:
                costs_20.append(0)          # Sucesso! Cliente bom passou.
        
        export_df['cost_threshold_0.20'] = costs_20
        export_df['decision_20'] = np.where(pred_20 == 1, 'Bloquear', 'Aprovar')

        # Cenário C: Modelo Padrão (Threshold 0.50)
        pred_50 = (probs >= 0.50).astype(int)
        costs_50 = []
        for real, pred in zip(export_df['class'], pred_50):
            if real == 1 and pred == 0: costs_50.append(COST_FRAUD)
            elif real == 0 and pred == 1: costs_50.append(COST_BLOCK)
            else: costs_50.append(0)
        export_df['cost_threshold_0.50'] = costs_50

        # 5. Diagnóstico Antes de Salvar
        print("\n--- DIAGNÓSTICO DO DATASET ---")
        print(f"Total de Transações: {len(export_df)}")
        print(f"Distribuição por Período:\n{export_df['Periodo'].value_counts()}")
        print(f"Fraudes Totais na Amostra: {export_df['class'].sum()}")
        print("-------------------------------------------------------")

        # 6. Exporta CSV
        print("4. Salvando CSV formatado...")
        export_df.to_csv(output_path, index=False, sep=';', decimal=',')
        print(f"✅ SUCESSO! Arquivo pronto em: {output_path}")

    except Exception as e:
        print(f"❌ Erro Crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_data_for_business_simulation()