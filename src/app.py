import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# --- Configura Caminhos ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "../models/challenger_model.pkl")

# --- Variáveis Globais ---
model = None
THRESHOLD = 0.20  # Threshold Otimizado (Financeiro)

# --- Classe de Engenharia (Feature Factory) ---
class FraudFeatureEngineer:
    def transform(self, X):
        X = X.copy()
        
        # 1. Feature: Hora do Dia (0-23)
        # O modelo espera 'hour', derivado de 'time'
        X['hour'] = X['time'].apply(lambda x: np.floor(x / 3600) % 24)
        
        # 2. Feature: Is Night (Madrugada)
        # O modelo espera 'is_night'
        X['is_night'] = X['hour'].apply(lambda x: 1 if x <= 6 else 0)
        
        # 3. Transformação de Log no Amount
        # O modelo espera 'amount_log'
        X['amount_log'] = np.log(X['amount'] + 0.001)
        
        return X

# --- Contrato de Dados (Schema) ---
# O cliente (maquininha):
class TransactionRequest(BaseModel):
    time: float
    amount: float
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0

# --- Inicialização ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ARQUIVO NÃO ENCONTRADO: {MODEL_PATH}")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ Modelo Challenger carregado! Threshold: {THRESHOLD}")
            print(f"📊 Colunas esperadas: {len(model.feature_names_in_)}")
        except Exception as e:
            print(f"❌ Erro crítico ao carregar pickle: {e}")
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "active", "model": "RandomForest-Challenger"}

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Modelo Offline")

    try:
        # 1. Converte JSON para DataFrame
        input_data = transaction.model_dump()
        df_raw = pd.DataFrame([input_data])
        
        # 2. Feature Engineering (Cria hour, is_night, amount_log)
        engineer = FraudFeatureEngineer()
        df_enriched = engineer.transform(df_raw)
        
        # 3. Alinha a ordem exata do modelo treinado
        expected_cols = model.feature_names_in_
        df_final = df_enriched[expected_cols]
        
        # 4. Inferência
        # Pega a probabilidade da classe 1 (Fraude)
        proba = model.predict_proba(df_final)[0][1]
        
        # 5. Decisão de Negócio
        is_fraud = bool(proba >= THRESHOLD)
        decision = "BLOQUEAR" if is_fraud else "APROVAR"
        
        return {
            "transaction_id": "uuid-test",
            "probability": round(float(proba), 4),
            "threshold_applied": THRESHOLD,
            "prediction": int(is_fraud),
            "decision": decision,
            "risk_level": "CRITICAL" if proba > 0.8 else ("HIGH" if is_fraud else "LOW")
        }

    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Erro de Coluna Faltante: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)