import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# =============================================================================
# Configuração da Página e Estilo
# =============================================================================
st.set_page_config(layout="wide", page_title="Análise de Futebol com IA")

# Estilo CSS para melhorar o visual
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E90FF;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #4682B4;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 10px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Funções de Preparação de Dados (Cache para performance)
# =============================================================================

@st.cache_data
def carregar_dados(caminho_arquivo='dataset.csv'):
    """Carrega e prepara os dados iniciais."""
    df = pd.read_csv(caminho_arquivo)
    colunas = {'Date': 'date', 'Home': 'home', 'Away': 'away', 'HG': 'home_goal', 'AG': 'away_goal', 'AvgCH': 'avg_odds_h', 'AvgCD': 'avg_odds_d', 'AvgCA': 'avg_odds_a'}
    df = df[list(colunas.keys())].copy()
    df.rename(columns=colunas, inplace=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.sort_values('date', inplace=True)
    df.dropna(inplace=True)
    return df

@st.cache_data
def engenharia_de_atributos(df, janela=10):
    """Cria features de performance baseadas numa janela de jogos."""
    df['resultado'] = np.select([(df['home_goal'] > df['away_goal']), (df['home_goal'] == df['away_goal'])], [0, 1], default=2)
    df['total_goals'] = df['home_goal'] + df['away_goal']
    
    df_jogos_list = []
    for _, row in df.iterrows():
        home_points = 3 if row['resultado'] == 0 else 1 if row['resultado'] == 1 else 0
        away_points = 3 if row['resultado'] == 2 else 1 if row['resultado'] == 1 else 0
        df_jogos_list.extend([
            {'date': row['date'], 'time': row['home'], 'gols_marcados': row['home_goal'], 'gols_sofridos': row['away_goal'], 'pontos': home_points},
            {'date': row['date'], 'time': row['away'], 'gols_marcados': row['away_goal'], 'gols_sofridos': row['home_goal'], 'pontos': away_points}
        ])
    df_jogos = pd.DataFrame(df_jogos_list).sort_values('date')

    stats_df = df_jogos.groupby('time').rolling(window=janela, on='date', min_periods=1).mean(numeric_only=True).shift(1)
    stats_df.rename(columns={'gols_marcados': 'media_gm', 'gols_sofridos': 'media_gs', 'pontos': 'media_pontos'}, inplace=True)
    stats_df = stats_df.reset_index()

    df_final = df.merge(stats_df, left_on=['date', 'home'], right_on=['date', 'time'], how='left').rename(columns={'media_gm': 'home_media_gm', 'media_gs': 'home_media_gs', 'media_pontos': 'home_media_pontos'}).drop('time', axis=1)
    df_final = df_final.merge(stats_df, left_on=['date', 'away'], right_on=['date', 'time'], how='left').rename(columns={'media_gm': 'away_media_gm', 'media_gs': 'away_media_gs', 'media_pontos': 'away_media_pontos'}).drop('time', axis=1)
    df_final.fillna(0, inplace=True)
    return df_final

# =============================================================================
# Funções de Visualização dos Modelos
# =============================================================================

def visualizar_classificacao(df, model):
    st.markdown('<p class="section-header">Classificação (XGBoost): Prever o Resultado do Jogo</p>', unsafe_allow_html=True)
    st.write("Este modelo prevê se o resultado será **Vitória da Casa**, **Empate** ou **Vitória do Visitante**.")
    
    features = ['home_media_pontos', 'away_media_pontos', 'home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
    target = 'resultado'
    _, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Acurácia do Modelo", value=f"{accuracy:.2%}")
    st.info(f"Isto significa que o modelo acerta o resultado do jogo em aproximadamente **{int(accuracy*100)}%** das vezes no conjunto de teste.")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Casa', 'Empate', 'Visitante'], yticklabels=['Casa', 'Empate', 'Visitante'], ax=ax)
    ax.set_title('Matriz de Confusão')
    ax.set_ylabel('Resultado Real')
    ax.set_xlabel('Resultado Previsto')
    st.pyplot(fig)

def visualizar_regressao(df, model):
    st.markdown('<p class="section-header">Regressão (XGBoost): Prever o Total de Golos</p>', unsafe_allow_html=True)
    st.write("Este modelo tenta prever o **número total de golos** numa partida.")
    
    features = ['home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
    target = 'total_goals'
    _, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.metric(label="Erro Médio (RMSE)", value=f"{rmse:.2f} golos")
    st.info(f"Isto significa que, em média, as previsões de total de golos do modelo erram por **{rmse:.2f}** golos para mais ou para menos.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, label='Previsões')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Linha Perfeita')
    ax.set_xlabel("Golos Reais")
    ax.set_ylabel("Golos Previstos")
    ax.set_title("Previsão de Golos vs. Realidade")
    ax.legend()
    st.pyplot(fig)

def visualizar_clustering(df, kmeans, scaler):
    st.markdown('<p class="section-header">Clustering (K-Means): Agrupar Equipas por Estilo</p>', unsafe_allow_html=True)
    st.write("Este modelo agrupa as equipas em 'clusters' com base na sua performance ofensiva e defensiva média ao longo do tempo.")

    team_stats = df.groupby('home').agg({'home_media_gm': 'mean', 'home_media_gs': 'mean'}).rename(columns={'home_media_gm': 'media_gols_marcados', 'home_media_gs': 'media_gols_sofridos'})
    team_stats_scaled = scaler.transform(team_stats)
    team_stats['cluster'] = kmeans.predict(team_stats_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=team_stats, x='media_gols_marcados', y='media_gols_sofridos', hue='cluster', palette='viridis', s=100, ax=ax)
    ax.set_title("Clusters de Equipas por Performance")
    ax.set_xlabel("Média de Golos Marcados (Força Ofensiva)")
    ax.set_ylabel("Média de Golos Sofridos (Fraqueza Defensiva)")
    st.pyplot(fig)
    
    st.write("Exemplos de equipas em cada cluster:")
    st.dataframe(team_stats.sort_values('cluster'))

def visualizar_mlp(df, model, scaler):
    st.markdown('<p class="section-header">Rede Neural (MLP): Previsão de Resultado</p>', unsafe_allow_html=True)
    st.write("Um modelo de rede neural leve para prever o resultado do jogo.")

    features = ['home_media_pontos', 'away_media_pontos', 'home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
    target = 'resultado'
    _, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Acurácia do Modelo MLP", value=f"{accuracy:.2%}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Casa', 'Empate', 'Visitante'], yticklabels=['Casa', 'Empate', 'Visitante'], ax=ax)
    ax.set_title('Matriz de Confusão')
    ax.set_ylabel('Resultado Real')
    ax.set_xlabel('Resultado Previsto')
    st.pyplot(fig)

def visualizar_previsoes(df, mlp_model, mlp_scaler, reg_model):
    st.markdown('<p class="section-header">Previsões para Próximos Jogos</p>', unsafe_allow_html=True)
    st.write("Selecione duas equipas para simular um confronto e ver as previsões dos modelos.")

    todas_as_equipas = sorted(pd.concat([df['home'], df['away']]).unique())
    
    col1, col2 = st.columns(2)
    with col1:
        time_casa = st.selectbox("Escolha o Time da Casa:", todas_as_equipas, index=0)
    with col2:
        opcoes_visitante = [t for t in todas_as_equipas if t != time_casa]
        time_visitante = st.selectbox("Escolha o Time Visitante:", opcoes_visitante, index=1)

    odd_h, odd_d, odd_a = df['avg_odds_h'].mean(), df['avg_odds_d'].mean(), df['avg_odds_a'].mean()

    if st.button("Gerar Previsão", type="primary"):
        home_stats = df[(df['home'] == time_casa) | (df['away'] == time_casa)].tail(1)
        away_stats = df[(df['home'] == time_visitante) | (df['away'] == time_visitante)].tail(1)

        home_latest_features = {'pontos': home_stats['home_media_pontos'].iloc[0] if time_casa == home_stats['home'].iloc[0] else home_stats['away_media_pontos'].iloc[0], 'gm': home_stats['home_media_gm'].iloc[0] if time_casa == home_stats['home'].iloc[0] else home_stats['away_media_gm'].iloc[0], 'gs': home_stats['home_media_gs'].iloc[0] if time_casa == home_stats['home'].iloc[0] else home_stats['away_media_gs'].iloc[0]}
        away_latest_features = {'pontos': away_stats['home_media_pontos'].iloc[0] if time_visitante == away_stats['home'].iloc[0] else away_stats['away_media_pontos'].iloc[0], 'gm': away_stats['home_media_gm'].iloc[0] if time_visitante == away_stats['home'].iloc[0] else away_stats['away_media_gm'].iloc[0], 'gs': away_stats['home_media_gs'].iloc[0] if time_visitante == away_stats['home'].iloc[0] else away_stats['away_media_gs'].iloc[0]}

        dados_previsao = pd.DataFrame([{'home_media_pontos': home_latest_features['pontos'], 'away_media_pontos': away_latest_features['pontos'], 'home_media_gm': home_latest_features['gm'], 'away_media_gm': away_latest_features['gm'], 'home_media_gs': home_latest_features['gs'], 'away_media_gs': away_latest_features['gs'], 'avg_odds_h': odd_h, 'avg_odds_d': odd_d, 'avg_odds_a': odd_a}])
        
        # Prepara os dados para os dois modelos
        features_clf = ['home_media_pontos', 'away_media_pontos', 'home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
        features_reg = ['home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
        
        # Normaliza os dados para o modelo MLP
        dados_previsao_scaled = mlp_scaler.transform(dados_previsao[features_clf])

        # --- Fazer Previsões ---
        # Classificação com Rede Neural (MLP)
        probabilidades = mlp_model.predict_proba(dados_previsao_scaled)
        prob_casa = probabilidades[0][0]
        prob_empate = probabilidades[0][1]
        prob_visitante = probabilidades[0][2]

        # Regressão com XGBoost
        total_golos_previsto = reg_model.predict(dados_previsao[features_reg])[0]

        # --- Exibir Resultados ---
        st.subheader("Resultados da Previsão")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Vitória {time_casa}", f"{prob_casa:.1%}")
        col2.metric("Empate", f"{prob_empate:.1%}")
        col3.metric(f"Vitória {time_visitante}", f"{prob_visitante:.1%}")
        st.metric("Previsão de Total de Golos", f"{total_golos_previsto:.2f}")

# =============================================================================
# Interface da Aplicação Streamlit
# =============================================================================

st.markdown('<p class="main-header">Análise de Futebol com IA</p>', unsafe_allow_html=True)

try:
    clf_model = joblib.load('models/classificador_xgboost.joblib')
    reg_model = joblib.load('models/regressor_xgboost.joblib')
    kmeans_model = joblib.load('models/kmeans_cluster.joblib')
    cluster_scaler = joblib.load('models/scaler_cluster.joblib')
    mlp_model = joblib.load('models/mlp_model.joblib')
    mlp_scaler = joblib.load('models/scaler_mlp.joblib')
except FileNotFoundError:
    st.error("ERRO: Modelos não encontrados. Execute o script 'training.py' primeiro para treinar e guardar os modelos.")
    st.stop()

dados = carregar_dados()
if dados is not None:
    dados_com_features = engenharia_de_atributos(dados)

    st.sidebar.markdown('<p class="sidebar-header">Navegação</p>', unsafe_allow_html=True)
    st.sidebar.markdown("<h4>Análise de IA</h4>", unsafe_allow_html=True)
    opcoes_menu = ["Página Inicial", "Previsões para Próximos Jogos", "Classificação", "Regressão", "Clustering", "Rede Neural (MLP)"]
    modelo_escolhido = st.sidebar.radio("Escolha uma opção:", opcoes_menu, label_visibility="collapsed")

    with st.container(border=True):
        if modelo_escolhido == "Página Inicial":
            st.markdown('<p class="section-header">Bem-vindo!</p>', unsafe_allow_html=True)
            st.image("https://images.pexels.com/photos/270085/pexels-photo-270085.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", caption="Análise de Dados no Futebol")
            st.write("""
            Esta aplicação demonstra como diferentes técnicas de Inteligência Artificial podem ser usadas para analisar e prever resultados de jogos de futebol.
            Use o menu na barra lateral para navegar entre as diferentes análises.
            """)
            st.subheader("Dados Utilizados (Amostra)")
            st.dataframe(dados_com_features.head())

        elif modelo_escolhido == "Previsões para Próximos Jogos":
            visualizar_previsoes(dados_com_features, mlp_model, mlp_scaler, reg_model)
        elif modelo_escolhido == "Classificação":
            visualizar_classificacao(dados_com_features, clf_model)
        elif modelo_escolhido == "Regressão":
            visualizar_regressao(dados_com_features, reg_model)
        elif modelo_escolhido == "Clustering":
            visualizar_clustering(dados_com_features, kmeans_model, cluster_scaler)
        elif modelo_escolhido == "Rede Neural (MLP)":
            visualizar_mlp(dados_com_features, mlp_model, mlp_scaler)
else:
    st.error("Não foi possível carregar os dados. Verifique o ficheiro 'dataset.csv'.")
