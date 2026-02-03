# 🛡️ Credit Card Fraud Detection System

## 📌 Visão Geral do Projeto
Este projeto implementa um pipeline completo de Data Science para detecção de fraudes em transações financeiras. O sistema foi desenhado sob a perspectiva de **Prevenção de Perdas (Loss Prevention)**, equilibrando a precisão técnica com restrições operacionais e a robustez contra *Data Leakage*.

O diferencial deste projeto é a aplicação de técnicas avançadas de validação temporal e foco em métricas de negócio (Recall vs Precisão), simulando um ambiente real de produção bancária.

## 📈 1. Entendimento do Problema de Negócio
O objetivo central é reduzir a taxa de *chargeback* (contestação de compra) sem elevar o atrito com clientes legítimos.

* **KPI Primário:** Recall (Taxa de Detecção de Fraude) - *Pegar o máximo de fraudes possível.*
* **KPI Secundário:** False Positive Rate (FPR) - *Evitar bloqueios indevidos.*
* **Restrição Operacional:** Capacidade de revisão manual limitada a 50 casos/dia.
* **SLA Técnico:** Latência < 100ms para decisão em tempo real.

## 🏗️ 2. Arquitetura do Pipeline de Dados
Seguimos uma arquitetura em camadas para garantir governança e reprodutibilidade:

1.  **Bronze (Raw):** Dados brutos ingeridos via Kaggle API.
2.  **Silver (Padronizada):** Conversão para **Parquet** (performance e tipagem) e padronização de schema (snake_case).
3.  **Trusted (Cleaned & Split):** * Deduplicação rigorosa (remoção de *retries* de sistema).
    * **Split Temporal:** Separação Treino/Teste respeitando a cronologia (Passado vs Futuro) para evitar *Look-ahead Bias*.

## 🔍 3. Principais Insights de Dados (EDA)
A Análise Exploratória foi realizada estritamente nos dados de treino para evitar vazamento de dados (*Data Leakage*). Principais descobertas:

* **Mito do Valor:** Fraudes não ocorrem apenas em valores altos. A distribuição de `Amount` em fraudes se sobrepõe às transações legítimas (testes de cartão e tickets médios).
* **Padrão Temporal:** "O crime não dorme". Enquanto transações legítimas caem 90% na madrugada, o volume de fraudes se mantém constante, aumentando o risco relativo nesse horário.
* **Assinatura Digital (Top Features):** As variáveis `V17`, `V14` e `V12` demonstraram altíssima capacidade discriminante. Valores negativos extremos nessas variáveis são fortes indicadores de atividade ilícita.

## 🛠️ Tecnologias e Ferramentas
* **Linguagem:** Python 3.12+
* **Manipulação:** Pandas, NumPy, PyArrow
* **Visualização:** Seaborn, Matplotlib
* **Ambiente:** Virtualenv
* **Versionamento:** Git & DVC (Data Version Control - *Planejado*)

## 📂 Estrutura do Projeto
```text
├── data/
│   ├── raw/          # Dados brutos (Imutáveis)
│   ├── silver/       # Parquet padronizado
│   └── trusted/      # Dados limpos e divididos (train/test)
├── reports/
│   └── figures/      # Gráficos gerados pela EDA (Comportamento e Features)
├── src/
│   ├── clean_data.py    # Limpeza, Deduplicação e Split Temporal
│   ├── eda_analysis.py  # Análise Exploratória e Geração de Insights
│   ├── ingest_silver.py # Ingestão para Silver
│   └── profiling.py     # Check de saúde dos dados
├── requirements.txt
└── README.md

🚀 Como Executar

    Instale as dependências:
    Bash

    pip install -r requirements.txt

    Pipeline de Dados (Ordem de Execução):
    Bash

    # 1. Ingestão e Padronização
    python src/ingest_silver.py

    # 2. Limpeza e Split Temporal (Gera a camada Trusted)
    python src/clean_data.py

    # 3. Geração de Relatórios e Gráficos (EDA)
    python src/eda_analysis.py

Autor

Rodrigo Neves