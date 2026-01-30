# 🛡️ Credit Card Fraud Detection System
## 📌 Visão Geral do Projeto
Este projeto implementa um pipeline completo de Data Science para detecção de fraudes em transações financeiras. O sistema foi desenhado sob a perspectiva de Prevenção de Perdas (Loss Prevention), equilibrando a precisão técnica com as restrições operacionais e de experiência do usuário (UX).

## 📈 1. Entendimento do Problema de Negócio
O objetivo central é reduzir a taxa de chargeback sem elevar o atrito com clientes legítimos.

KPI Primário: Recall (Taxa de Detecção de Fraude).

KPI Secundário: False Positive Rate (FPR) para minimizar bloqueios indevidos.

Restrição Operacional: O time de revisão manual tem capacidade para apenas 50 casos/dia.

SLA Técnico: Tempo de resposta do modelo deve ser < 100ms para integração em tempo real.


## 🏗️ 2. Arquitetura do Pipeline de Dados
Seguimos a Medallion Architecture para garantir linhagem e governança dos dados:

Bronze (Raw): Dados brutos ingeridos via Kaggle API, mantendo a integridade original.

Silver (Trusted): Dados convertidos para Apache Parquet via pyarrow. Nesta etapa, aplicamos padronização de schemas (snake_case) e garantimos a tipagem forte das variáveis.

Gold (Processed): (Próxima etapa) Dados limpos e enriquecidos prontos para o treinamento do modelo.


## 🔍 3. Mapeamento e Diagnóstico (EDA Inicial)
Durante o profiling inicial dos dados (Fase 2), identificamos pontos críticos para a estratégia de modelagem:

Extremo Desbalanceamento: Apenas 0.1727% das transações são fraudulentas (492 casos em 284.807).

Janela Temporal: O dataset cobre 48 horas de transações.

Privacidade (LGPD): Dados anonimizados via PCA para proteção de PII (Personally Identifiable Information).


## 🛠️ Tecnologias e Ferramentas

Linguagem: Python 3.12+.

Manipulação de Dados: Pandas & PyArrow.

Ingestão: Kaggle API (Extração Automática).

Documentação: Notion & Miro (Design Doc).

## 📂 Estrutura do Projeto
```text
├── data/
│   ├── raw/          # Dados brutos (Imutáveis)
│   ├── silver/       # Dados padronizados em Parquet
├── src/
│   └── data/         # Scripts de processamento de dados
│       ├── ingestion.py          # Download via API
│       ├── ingestion_silver.py   # Padronização e conversão
│       └── profiling.py          # Relatório de saúde dos dados
├── requirements.txt  # Dependências do projeto
└── README.md


## 🚀 Como Executar
Configure suas credenciais do Kaggle nas variáveis de ambiente.

Instale as dependências: pip install -r requirements.txt.

Execute o pipeline de ingestão: python src/data/ingestion_silver.py.

## Autor
Rodrigo Neves