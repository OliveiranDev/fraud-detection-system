import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_baseline():
    base_path = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base_path, "../data/gold/")
    models_path = os.path.join(base_path, "../models/")
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        
    print("--- INICIANDO TREINAMENTO BASELINE ---")
    
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
    
    print(f"Features selecionadas: {X_train.columns.tolist()[-5:]} ...") # Ver as últimas para checar feature eng

    # 3. Pipeline de Treinamento
    # Verifica Regressão Logística que precisa de dados na mesma escala (StandardScaler)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    
    print("Treinando Regressão Logística...")
    pipeline.fit(X_train, y_train)
    
    # 4. Avaliação
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- PERFORMANCE DO BASELINE ---")
    print(classification_report(y_test, y_pred))
    
    # Métricas Específicas de Fraude
    roc_auc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUPRC (Foco em Desbalanceados): {auprc:.4f}")
    
    # Matriz de Confusão Visual
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nResumo de Negócio:")
    print(f"✅ Fraudes Detectadas (TP): {tp}")
    print(f"❌ Fraudes Perdidas (FN): {fn}")
    print(f"⚠️ Clientes Bloqueados Indevidamente (FP): {fp}")
    
    # Salva Modelo
    model_file = os.path.join(models_path, "baseline_model.pkl")
    joblib.dump(pipeline, model_file)
    print(f"\nModelo salvo em: {model_file}")

if __name__ == "__main__":
    train_baseline()