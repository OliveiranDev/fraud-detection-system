import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp

def check_data_drift():
    print("--- 🔍 INICIANDO MONITORAMENTO DE DRIFT (CUSTOM KS-TEST) ---")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Reference: O dado que o modelo aprendeu (Treino)
    train_path = os.path.join(base_path, "../data/gold/train_data.parquet")
    # Current: O dado que está chegando (Teste/Produção)
    prod_path = os.path.join(base_path, "../data/gold/test_data.parquet")
    
    try:
        # 1. Carrega Dados
        ref_df = pd.read_parquet(train_path)
        curr_df = pd.read_parquet(prod_path)
        
        # Simula uma janela de produção (50% do teste aleatoriamente)
        curr_sample = curr_df.sample(frac=0.5, random_state=42)
        
        # 2. Definir Features Críticas para Monitorar (monitora somente dados importantes)
        features_to_monitor = ['amount_log', 'is_night', 'v14', 'v17', 'v12', 'v4', 'v11']
        
        print(f"\nComparando Distribuições: Treino ({len(ref_df)} linhas) vs Produção ({len(curr_sample)} linhas)")
        print(f"Teste Estatístico: Kolmogorov-Smirnov (KS-Test)")
        print(f"Limiar de Alerta (P-Value): < 0.05 (Confiança de 95%)\n")
        
        print(f"{'FEATURE':<15} | {'DRIFT?':<10} | {'P-VALUE':<10} | {'STATUS'}")
        print("-" * 60)
        
        drift_count = 0
        
        for feature in features_to_monitor:
            # Pega as séries de dados
            data_ref = ref_df[feature].dropna()
            data_cur = curr_sample[feature].dropna()
            
            # Aplica Teste KS
            # Null Hypothesis (H0): As distribuições são iguais.
            # Se p_value < 0.05, rejeitamos H0 -> Ocorreu Drift.
            stat, p_value = ks_2samp(data_ref, data_cur)
            
            # Formata output
            is_drift = p_value < 0.05
            drift_status = "🔴 ALERTA" if is_drift else "✅ OK"
            if is_drift:
                drift_count += 1
            
            print(f"{feature:<15} | {str(is_drift):<10} | {p_value:.4f}     | {drift_status}")

        print("-" * 60)
        
        # 3. Veredito Final
        drift_ratio = drift_count / len(features_to_monitor)
        print(f"\n📊 RESUMO DO MONITORAMENTO:")
        print(f"Features com anomalia: {drift_count}/{len(features_to_monitor)} ({drift_ratio:.1%})")
        
        if drift_ratio > 0.3: # Se mais de 30% das features mudaram
            print("\n🚨 CONCLUSÃO: DRIFT CRÍTICO DETECTADO!")
            print("Ação: O comportamento dos dados mudou significativamente.")
            print("Recomendação: 1. Investigar a origem dos dados. 2. Retreinar o modelo (Etapa 11).")
        else:
            print("\n✅ CONCLUSÃO: SISTEMA ESTÁVEL.")
            print("Ação: Nenhuma intervenção necessária. O modelo segue saudável.")

    except Exception as e:
        print(f"❌ Erro crítico no monitoramento: {e}")

if __name__ == "__main__":
    check_data_drift()