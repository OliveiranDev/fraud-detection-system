import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix

def evaluate_financial_impact():
    base_path = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base_path, "../data/gold/")
    models_path = os.path.join(base_path, "../models/")
    reports_path = os.path.join(base_path, "../reports/figures/")
    
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    print("--- INICIANDO AVALIAÇÃO DE NEGÓCIO ---")
    
    # 1. Carrega Modelo Challenger e Dados de Teste
    try:
        model = joblib.load(os.path.join(models_path, "challenger_model.pkl"))
        test_df = pd.read_parquet(os.path.join(gold_path, "test_data.parquet"))
    except Exception as e:
        print(f"❌ Erro ao carregar artefatos: {e}")
        return

    # Prepara dados
    target = 'class'
    drop_cols = [target, 'time']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]
    
    # 2. Obter Probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 3. Simulação Financeira (Loop de Thresholds)
    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = []
    
    # Premissas de Negócio
    COST_FN = 100  # Perda média por fraude não pega (Chargeback)
    COST_FP = 2    # Custo de fricção (SMS, Call Center, Risco de Churn)
    
    print(f"--- PREMISSAS ---")
    print(f"Custo Fraude (FN): €{COST_FN} | Custo Bloqueio Indevido (FP): €{COST_FP}")
    
    best_threshold = 0.5
    min_cost = float('inf')
    
    for thresh in thresholds:
        y_pred_t = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        
        # Fórmula do Custo Total
        total_cost = (fn * COST_FN) + (fp * COST_FP)
        costs.append(total_cost)
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = thresh

    # 4. Resultados
    print(f"\n--- RESULTADO DA OTIMIZAÇÃO ---")
    
    # Compara Padrão vs Otimizado
    y_pred_default = model.predict(X_test) # Threshold 0.5
    cm_def = confusion_matrix(y_test, y_pred_default)
    cost_default = (cm_def.ravel()[2] * COST_FN) + (cm_def.ravel()[1] * COST_FP)
    
    print(f"Threshold Padrão (0.50): Custo Total = €{cost_default:,.2f}")
    print(f"Threshold Otimizado ({best_threshold:.2f}): Custo Total = €{min_cost:,.2f}")
    
    savings = cost_default - min_cost
    print(f"💰 Economia Projetada: €{savings:,.2f} (Melhora de {savings/cost_default:.1%})")
    
    # 5. Gera Gráfico "Money Plot"
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, color='red', label='Prejuízo Total')
    plt.axvline(best_threshold, color='green', linestyle='--', label=f'Otimizado ({best_threshold:.2f})')
    plt.axvline(0.5, color='gray', linestyle=':', label='Padrão (0.5)')
    
    plt.title('Curva de Custo Financeiro x Threshold de Decisão')
    plt.xlabel('Threshold de Probabilidade')
    plt.ylabel('Custo Total Estimado (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(reports_path, "8_financial_impact_analysis.png")
    plt.savefig(plot_file)
    print(f"✅ Gráfico de impacto financeiro salvo em: {plot_file}")
    
    # 6. Salva Métricas Finais do Otimizado
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm_opt.ravel()
    
    print(f"\nMatriz de Confusão (Threshold {best_threshold:.2f}):")
    print(f"TP (Fraudes Pegas): {tp}")
    print(f"FN (Fraudes Perdidas): {fn}")
    print(f"FP (Bloqueios Indevidos): {fp}")

if __name__ == "__main__":
    evaluate_financial_impact()