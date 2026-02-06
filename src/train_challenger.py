import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_challenger():
    base_path = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base_path, "../data/gold/")
    models_path = os.path.join(base_path, "../models/")
    
    print("--- INICIANDO TREINAMENTO CHALLENGER (RANDOM FOREST) ---")
    
    # 1. Carregar Dados Gold
    try:
        train_df = pd.read_parquet(os.path.join(gold_path, "train_data.parquet"))
        test_df = pd.read_parquet(os.path.join(gold_path, "test_data.parquet"))
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # 2. Separar Features e Target
    target = 'class'
    drop_cols = [target, 'time'] 
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]
    
    # 3. Pipeline Challenger
    # n_jobs=-1 usa todos os núcleos do processador
    # class_weight='balanced' continua sendo vital
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=10 # Limita profundidade para evitar overfitting
    )
    
    print("Treinando Random Forest (pode levar alguns segundos)...")
    model.fit(X_train, y_train)
    
    # 4. Avaliação
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- PERFORMANCE DO CHALLENGER ---")
    print(classification_report(y_test, y_pred))
    
    # Métricas de Negócio
    roc_auc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    print(f"\nResumo de Negócio (Challenger):")
    print(f"✅ Fraudes Detectadas (TP): {tp} (Baseline era 69)")
    print(f"❌ Fraudes Perdidas (FN): {fn} (Baseline era 5)")
    print(f"📉 Clientes Bloqueados Indevidamente (FP): {fp} (Baseline era 2783)")
    
    # Comparativo Rápido
    reduction_fp = 2783 - fp
    if reduction_fp > 0:
        print(f"\nSUCESSO: Deixamos de bloquear {reduction_fp} clientes legítimos!")
    
    # Salvar Modelo
    model_file = os.path.join(models_path, "challenger_model.pkl")
    joblib.dump(model, model_file)
    print(f"Modelo salvo em: {model_file}")

if __name__ == "__main__":
    train_challenger()