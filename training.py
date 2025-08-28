import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

# --- FUNÇÕES DE PREPARAÇÃO DE DADOS ---

def carregar_dados(caminho_arquivo='dataset.csv'):
    """Carrega e prepara os dados iniciais."""
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO: O ficheiro de dados '{caminho_arquivo}' não foi encontrado.")
        print("Por favor, crie a pasta 'data' e coloque o seu 'dataset.csv' dentro dela.")
        return None
        
    print(f"Carregando dados do arquivo local: {caminho_arquivo}")
    df = pd.read_csv(caminho_arquivo)
    colunas = {'Date': 'date', 'Home': 'home', 'Away': 'away', 'HG': 'home_goal', 'AG': 'away_goal', 'AvgCH': 'avg_odds_h', 'AvgCD': 'avg_odds_d', 'AvgCA': 'avg_odds_a'}
    
    if not all(col in df.columns for col in colunas.keys()):
        print("ERRO: O dataset não contém todas as colunas necessárias.")
        return None

    df = df[list(colunas.keys())].copy()
    df.rename(columns=colunas, inplace=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.sort_values('date', inplace=True)
    df.dropna(inplace=True)
    return df

def engenharia_de_atributos(df, janela=10):
    """Cria features de performance baseadas numa janela de jogos."""
    print("A criar features de engenharia de atributos...")
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

# --- FUNÇÕES DE TREINO ---

def treinar_modelo_classificacao(df):
    print("Treinando modelo de Classificação (XGBoost)...")
    features = ['home_media_pontos', 'away_media_pontos', 'home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
    target = 'resultado'
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', n_estimators=100)
    model.fit(X_train, y_train, verbose=False)

    # Imprime o resultado no terminal
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  -> Acurácia do XGBoost Classificador: {accuracy:.2%}")

    return model

def treinar_modelo_regressao(df):
    print("Treinando modelo de Regressão (XGBoost)...")
    features = ['home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
    target = 'total_goals'
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=100)
    model.fit(X_train, y_train, verbose=False)

    # Imprime o resultado no terminal
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  -> RMSE do XGBoost Regressor: {rmse:.2f} golos")

    return model

def treinar_modelo_clustering(df):
    print("Treinando modelo de Clustering (K-Means)...")
    team_stats = df.groupby('home').agg({'home_media_gm': 'mean', 'home_media_gs': 'mean'}).rename(columns={'home_media_gm': 'media_gols_marcados', 'home_media_gs': 'media_gols_sofridos'})
    scaler = StandardScaler()
    team_stats_scaled = scaler.fit_transform(team_stats)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=25)
    kmeans.fit(team_stats_scaled)
    print("  -> Modelo de Clustering treinado com sucesso.")
    return kmeans, scaler, team_stats

def treinar_modelo_mlp(df):
    print("Treinando modelo de Rede Neural (MLP)...")
    features = ['home_media_pontos', 'away_media_pontos', 'home_media_gm', 'away_media_gm', 'home_media_gs', 'away_media_gs', 'avg_odds_h', 'avg_odds_d', 'avg_odds_a']
    target = 'resultado'
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # AJUSTE: Adicionada uma camada extra e aumentado max_iter para tentar melhorar a performance.
    model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    model.fit(X_train_scaled, y_train)

    # Imprime o resultado no terminal
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  -> Acurácia da Rede Neural (MLP): {accuracy:.2%}")

    return model, scaler

# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')

    dados = carregar_dados()
    
    if dados is not None:
        dados_com_features = engenharia_de_atributos(dados)

        # Treina os modelos
        clf_model = treinar_modelo_classificacao(dados_com_features)
        reg_model = treinar_modelo_regressao(dados_com_features)
        kmeans_model, cluster_scaler, _ = treinar_modelo_clustering(dados_com_features)
        mlp_model, mlp_scaler = treinar_modelo_mlp(dados_com_features)

        # Salva os modelos
        print("\nSalvando modelos em disco...")
        joblib.dump(clf_model, 'models/classificador_xgboost.joblib')
        joblib.dump(reg_model, 'models/regressor_xgboost.joblib')
        joblib.dump(kmeans_model, 'models/kmeans_cluster.joblib')
        joblib.dump(cluster_scaler, 'models/scaler_cluster.joblib')
        joblib.dump(mlp_model, 'models/mlp_model.joblib')
        joblib.dump(mlp_scaler, 'models/scaler_mlp.joblib')
        
        print("\nTreino concluído! Todos os modelos foram guardados na pasta 'models/'.")
        print("Agora pode executar a aplicação com 'streamlit run app.py'.")