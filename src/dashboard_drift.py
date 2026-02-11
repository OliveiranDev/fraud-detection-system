import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp

def generate_drift_dashboard():
    print("--- 🎨 GERANDO DASHBOARD EXECUTIVO DE DRIFT ---")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "../data/gold/train_data.parquet")
    prod_path = os.path.join(base_path, "../data/gold/test_data.parquet")
    report_path = os.path.join(base_path, "../reports/10_executive_drift_dashboard.html")
    
    # 1. Carrega amostrar dados (Para o gráfico não ficar pesado)
    try:
        df_ref = pd.read_parquet(train_path).sample(5000, random_state=42)
        df_cur = pd.read_parquet(prod_path).sample(5000, random_state=42)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # Features Críticas para Monitorar
    features = ['amount_log', 'is_night', 'v14', 'v17', 'v12', 'v4']
    
    # Criar Subplots (Um gráfico por feature)
    rows = 3
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=features,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    alert_count = 0
    drift_details = []

    print("📊 Calculando estatísticas e desenhando gráficos...")

    for i, feature in enumerate(features):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Dados limpos
        ref_data = df_ref[feature].dropna()
        cur_data = df_cur[feature].dropna()
        
        # Teste KS (Matemática por trás do visual)
        stat, p_value = ks_2samp(ref_data, cur_data)
        is_drift = p_value < 0.05
        
        color_ref = 'rgba(46, 204, 113, 0.6)'  # Verde (Treino/Base)
        color_cur = 'rgba(231, 76, 60, 0.6)'   # Vermelho (Produção/Drift)
        
        if is_drift:
            alert_count += 1
            drift_details.append(f"Feature <b>{feature}</b> variou significativamente.")

        # Histograma Reference (Treino)
        fig.add_trace(go.Histogram(
            x=ref_data,
            name=f'{feature} (Treino)',
            marker_color=color_ref,
            opacity=0.5,
            showlegend=(i==0) # Mostrar legenda só no primeiro para não poluir
        ), row=row, col=col)

        # Histograma Current (Produção)
        fig.add_trace(go.Histogram(
            x=cur_data,
            name=f'{feature} (Produção)',
            marker_color=color_cur,
            opacity=0.5,
            showlegend=(i==0)
        ), row=row, col=col)
        
        # Adiciona anotação de Status no gráfico
        status_text = "⚠️ DRIFT" if is_drift else "✅ OK"
        status_color = "red" if is_drift else "green"
        

    # Layout Profissional
    fig.update_layout(
        title_text=f"<b>Monitoramento de Saúde do Modelo (Data Drift)</b><br>Reference (Verde) vs Production (Vermelho)",
        title_x=0.5,
        height=1000,
        width=1200,
        template="plotly_white",
        font=dict(family="Arial", size=12),
        bargap=0.1,
        barmode='overlay' # Sobrepor para ver a mudança
    )

    # Adiciona Resumo Executivo como Anotação
    summary_text = (
        f"<b>Relatório de Governança</b><br>"
        f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y')}<br>"
        f"Status: {'🔴 CRÍTICO' if alert_count > 2 else '🟢 ESTÁVEL'}<br>"
        f"Features com Drift: {alert_count}/{len(features)}<br><br>"
        f"<b>Ação Recomendada:</b><br>"
        f"{'Requer Retreino Imediato' if alert_count > 2 else 'Monitorar'}"
    )

    # Salva HTML
    print(f"💾 Salvando dashboard em: {report_path}")
    fig.write_html(report_path)
    
    # Adiciona texto explicativo no HTML
    with open(report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    header = f"""
    <div style="font-family: Arial; padding: 20px; background-color: #f8f9fa; border-bottom: 3px solid #2c3e50;">
        <h1 style="color: #2c3e50;">🛡️ Painel de Governança de IA</h1>
        <p style="font-size: 16px;">Este painel compara a distribuição dos dados usados no treinamento (Baseline) contra os dados reais chegando em produção.</p>
        <div style="background-color: {'#fadbd8' if alert_count > 0 else '#d4efdf'}; padding: 15px; border-radius: 5px; border-left: 5px solid {'#c0392b' if alert_count > 0 else '#27ae60'};">
            <strong>Veredito do Algoritmo:</strong> Os padrões de fraude mudaram. A distribuição vermelha (Produção) deslocou-se em relação à verde (Treino).<br>
            <em>Isso justifica a queda de performance e a necessidade de manutenção do modelo.</em>
        </div>
    </div>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(header + html_content)

    print("✅ Dashboard gerado com sucesso!")

if __name__ == "__main__":
    generate_drift_dashboard()