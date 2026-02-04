# 🛡️ Credit Card Fraud Detection System

## 📌 Visão Geral do Projeto
Este projeto implementa um pipeline completo de Data Science para detecção de fraudes em transações financeiras. O sistema foi desenhado sob a perspectiva de **Prevenção de Perdas (Loss Prevention)**, equilibrando a precisão técnica com restrições operacionais e a robustez contra *Data Leakage*.

O diferencial deste projeto é a aplicação de técnicas de validação temporal e foco em métricas de negócio (Recall vs Precisão), simulando um ambiente real de produção bancária.

## 📈 1. Entendimento do Problema de Negócio
O objetivo central é reduzir a taxa de *chargeback* (contestação de compra) sem elevar o atrito com clientes legítimos.

* **KPI Primário:** Recall (Taxa de Detecção de Fraude) - *Pegar o máximo de fraudes possível.*
* **KPI Secundário:** False Positive Rate (FPR) - *Evitar bloqueios indevidos.*
* **Restrição Operacional:** Capacidade de revisão manual limitada a 50 casos/dia.
* **SLA Técnico:** Latência < 100ms para decisão em tempo real.

## 🏗️ 2. Arquitetura do Pipeline de Dados
Segue uma arquitetura em camadas para garantir governança e reprodutibilidade:

1.  **Bronze (Raw):** Dados brutos ingeridos via Kaggle API.
2.  **Silver (Padronizada):** Conversão para **Parquet** (performance e tipagem) e padronização de schema (snake_case).
3.  **Trusted (Cleaned & Split):** * Deduplicação rigorosa (remoção de *retries* de sistema).
    * **Split Temporal:** Separação Treino/Teste respeitando a cronologia (Passado vs Futuro) para evitar *Look-ahead Bias*.

## 🔍 3. Principais Insights de Dados (EDA)
A Análise Exploratória foi realizada estritamente nos dados de treino para evitar vazamento de dados (*Data Leakage*). Principais descobertas:

* **Mito do Valor:** Fraudes não ocorrem apenas em valores altos. A distribuição de `Amount` em fraudes se sobrepõe às transações legítimas (testes de cartão e tickets médios).
* **Padrão Temporal:** "O crime não dorme". Enquanto transações legítimas caem 90% na madrugada, o volume de fraudes se mantém constante, aumentando o risco relativo nesse horário.
* **Assinatura Digital (Top Features):** As variáveis `V17`, `V14` e `V12` demonstraram altíssima capacidade discriminante. Valores negativos extremos nessas variáveis são fortes indicadores de atividade ilícita.

## ⚙️ 4. Feature Engineering (Camada Gold)
Essa etapa traduz os insights de negócio da EDA em vetores matemáticos para o modelo, criando a **Feature Store** na pasta `data/gold/`.

* **Arquitetura:** Implementa uma classe customizada `FraudFeatureEngineer` (herdando de `sklearn.base.TransformerMixin`). Isso garante que o mesmo pipeline de transformação usado no treino seja reutilizado em produção via API, eliminando *Training-Serving Skew*.
* **Novas Features Criadas:**
    1.  `is_night` (Binária): Penaliza transações realizadas na madrugada (janela de risco identificada na EDA).
    2.  `amount_log` (Float): Normalização logarítmica do valor da transação ($\ln(x + 0.001)$) para reduzir a variância de dados financeiros com distribuição *Power Law*.
    3.  `hour` (Int): Ciclo de 24h derivado do timestamp absoluto.

## 🛠️ Tecnologias e Ferramentas
* **Linguagem:** Python 3.12+
* **Manipulação:** Pandas, NumPy, PyArrow
* **Machine Learning:** Scikit-Learn (Pipelines & Transformers)
* **Visualização:** Seaborn, Matplotlib
* **Ambiente:** Virtualenv
* **Versionamento:** Git & DVC (Data Version Control - *Planejado*)

## 📂 Estrutura do Projeto
```text
├── data/
│   ├── raw/          # Dados brutos (Imutáveis)
│   ├── silver/       # Parquet padronizado
│   ├── trusted/      # Dados limpos e divididos (train/test)
│   └── gold/         # Feature Store (Dados enriquecidos prontos para modelo)
├── reports/
│   └── figures/      # Gráficos gerados pela EDA
├── src/
│   ├── clean_data.py          # Limpeza e Split Temporal
│   ├── eda_analysis.py        # Análise Exploratória
│   ├── feature_engineering.py # Transformação de Features (Scikit-Learn)
│   ├── ingest_silver.py       # Ingestão Inicial
│   └── profiling.py           # Check de saúde
├── requirements.txt
└── README.md

Como Executar

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Pipeline de Dados (Ordem de Execução):**
    ```bash
    # 1. Ingestão e Padronização
    python src/ingest_silver.py
    
    # 2. Limpeza e Split Temporal (Camada Trusted)
    python src/clean_data.py
    
    # 3. Geração de Insights (EDA)
    python src/eda_analysis.py
    
    # 4. Feature Engineering (Camada Gold)
    python src/feature_engineering.py
    ```

Autor
Rodrigo Neves