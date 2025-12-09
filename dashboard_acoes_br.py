import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Ações Brasileiras",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇧🇷 Dashboard Interativo - Top 30 Ações Brasileiras")
st.markdown("---")

# Top 30 tickers
tickers_br = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'BBAS3.SA', 'WEGE3.SA', 'RENT3.SA', 'SUZB3.SA', 'RAIL3.SA',
    'JBSS3.SA', 'MGLU3.SA', 'LREN3.SA', 'GGBR4.SA', 'EMBR3.SA',
    'RADL3.SA', 'VIVT3.SA', 'ELET3.SA', 'CMIG4.SA', 'CSAN3.SA',
    'PRIO3.SA', 'KLBN11.SA', 'HAPV3.SA', 'BPAC11.SA', 'ASAI3.SA',
    'CYRE3.SA', 'EQTL3.SA', 'TOTS3.SA', 'BEEF3.SA', 'PETZ3.SA'
]

setores = {
    'PETR4.SA': 'Petróleo e Gás', 'VALE3.SA': 'Mineração', 'ITUB4.SA': 'Bancos',
    'BBDC4.SA': 'Bancos', 'ABEV3.SA': 'Bebidas', 'BBAS3.SA': 'Bancos',
    'WEGE3.SA': 'Máquinas', 'RENT3.SA': 'Locação', 'SUZB3.SA': 'Papel e Celulose',
    'RAIL3.SA': 'Logística', 'JBSS3.SA': 'Alimentos', 'MGLU3.SA': 'Varejo',
    'LREN3.SA': 'Varejo', 'GGBR4.SA': 'Siderurgia', 'EMBR3.SA': 'Aeronáutica',
    'RADL3.SA': 'Farmacêutico', 'VIVT3.SA': 'Telecom', 'ELET3.SA': 'Energia',
    'CMIG4.SA': 'Energia', 'CSAN3.SA': 'Energia', 'PRIO3.SA': 'Petróleo',
    'KLBN11.SA': 'Papel e Celulose', 'HAPV3.SA': 'Saúde', 'BPAC11.SA': 'Bancos',
    'ASAI3.SA': 'Varejo', 'CYRE3.SA': 'Construção', 'EQTL3.SA': 'Energia',
    'TOTS3.SA': 'Educação', 'BEEF3.SA': 'Alimentos', 'PETZ3.SA': 'Varejo'
}

@st.cache_data(ttl=3600)  # Cache por 1 hora para dados frescos
def carregar_dados():
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=365)
    dados_acoes = {}
    precos_fechamento = pd.DataFrame()

    for ticker in tickers_br:
        try:
            acao = yf.Ticker(ticker)
            hist = acao.history(start=data_inicio, end=data_fim)
            if not hist.empty:
                dados_acoes[ticker] = {'historico': hist, 'setor': setores.get(ticker, 'Outros')}
                precos_fechamento[ticker] = hist['Close']
        except:
            pass

    # Calcular métricas
    metricas = []
    for ticker, dados in dados_acoes.items():
        hist = dados['historico']
        retornos_diarios = hist['Close'].pct_change().dropna()
        retorno_total = ((hist['Close'][-1] / hist['Close'][0]) - 1) * 100
        retorno_anualizado = ((1 + retorno_total/100) ** (252/len(hist))) - 1 * 100
        volatilidade = retornos_diarios.std() * np.sqrt(252) * 100
        sharpe = (retorno_anualizado - 12) / volatilidade if volatilidade > 0 else 0
        cumulative = (1 + retornos_diarios).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        max_drawdown = drawdown.min()

        # Beta simples (vs PETR4 como proxy de mercado)
        beta = 1.0  # Default
        if ticker != 'PETR4.SA' and 'PETR4.SA' in precos_fechamento.columns:
            ret_merc = precos_fechamento['PETR4.SA'].pct_change().dropna()
            ret_acao = precos_fechamento[ticker].pct_change().dropna()
            ret_comum = pd.concat([ret_acao, ret_merc], axis=1).dropna()
            if len(ret_comum) > 30:
                beta = ret_comum.cov().iloc[0,1] / ret_comum.iloc[:,1].var()

        risco = 'Baixo' if volatilidade < 20 else 'Médio' if volatilidade < 35 else 'Alto'

        metricas.append({
            'Ticker': ticker.replace('.SA', ''),
            'Setor': dados['setor'],
            'Preço Atual': hist['Close'][-1],
            'Retorno Anual (%)': retorno_anualizado,
            'Volatilidade (%)': volatilidade,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_drawdown,
            'Beta': beta,
            'Risco': risco,
            'Volume Médio': hist['Volume'].mean()
        })

    df_metricas = pd.DataFrame(metricas).sort_values('Retorno Anual (%)', ascending=False)
    correlacao = precos_fechamento.pct_change().corr()
    return df_metricas, correlacao, dados_acoes, precos_fechamento

df_metricas, correlacao, dados_acoes, precos = carregar_dados()

# Sidebar para filtros
st.sidebar.header("Filtros")
setor_selecionado = st.sidebar.multiselect("Filtrar por Setor:", options=df_metricas['Setor'].unique(), default=df_metricas['Setor'].unique())
df_filtrado = df_metricas[df_metricas['Setor'].isin(setor_selecionado)]

# Métricas gerais
col1, col2, col3, col4 = st.columns(4)
col1.metric("Retorno Médio Anual", f"{df_filtrado['Retorno Anual (%)'].mean():.2f}%")
col2.metric("Volatilidade Média", f"{df_filtrado['Volatilidade (%)'].mean():.2f}%")
col3.metric("Sharpe Médio", f"{df_filtrado['Sharpe Ratio'].mean():.2f}")
col4.metric("Ações Analisadas", len(df_filtrado))

st.markdown("---")

# Abas para navegação
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔗 Correlações", "📊 Setorial", "🎯 Recomendações"])

with tab1:
    # Gráficos de performance
    fig_perf = make_subplots(rows=2, cols=2, subplot_titles=('Top 10 Retorno', 'Risco', 'Risco-Retorno', 'Drawdown'))

    # Top 10 Retorno
    top10 = df_filtrado.head(10)
    fig_perf.add_trace(go.Bar(x=top10['Ticker'], y=top10['Retorno Anual (%)'], name='Retorno'), row=1, col=1)

    # Distribuição Risco
    risco_counts = df_filtrado['Risco'].value_counts()
    fig_perf.add_trace(go.Pie(labels=risco_counts.index, values=risco_counts.values, name='Risco'), row=1, col=2)

