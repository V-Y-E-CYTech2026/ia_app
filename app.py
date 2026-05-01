import streamlit as st
import mlflow
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import os
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from omegaconf import OmegaConf
from fredapi import Fred
from datetime import datetime, timedelta

# --- CONFIGURATION ---
FRED_API_KEY = '88f89b6134c3f03d05307558eeeafcf7'

if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver("env", lambda var, default=None: os.getenv(var, default))

project_path = Path.cwd()
bootstrap_project(project_path)


# --- FONCTION LIVE DATA (INTEGRATION FRED) ---
@st.cache_data(ttl=3600)
def get_live_features(symbol):
    # On prend une marge pour le calcul des SMA 200 et du shift macro (30j)
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')

    # 1. Données Marché
    df = yf.download(symbol, start=start_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Indicateurs Techniques
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['real_logs'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Close'].pct_change().rolling(21).std()
    df['Z_Score_10'] = (df['Close'] - df['Close'].rolling(10).mean()) / df['Close'].rolling(10).std()

    for length in [10, 20, 50, 200]:
        df[f'SMA_{length}'] = ta.sma(df['Close'], length=length)

    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    df['Dist_SMA_10'] = (df['Close'] - df['SMA_10']) / df['SMA_10']
    df['Log_Volume'] = np.log(df['Volume'] + 1)
    df['Log_Vol_Change'] = df['Log_Volume'].diff().shift(1)
    df['Trend_Signal'] = (df['Close'] > df['SMA_10']).astype(int)
    df['Ret_L1'] = df['Close'].pct_change().shift(1)
    df['Ret_L2'] = df['Close'].pct_change().shift(2)

    vix = yf.download("^VIX", start=start_date)['Close']
    df['VIX'] = vix.shift(1)
    df['VIX_Change'] = df['VIX'].diff()

    # 2. Données Macro Réelles (FRED)
    try:
        fred = Fred(api_key=FRED_API_KEY)
        macro_map = {
            'yield_10y': 'DGS10',
            'yield_2y': 'DGS2',
            'Fed_Rate': 'FEDFUNDS',
            'Unemployment': 'UNRATE',
            'Inflation_Idx': 'CPIAUCSL',
            'stress_fed': 'STLFSI4'
        }

        macro_data = {}
        for col, fred_id in macro_map.items():
            macro_data[col] = fred.get_series(fred_id, observation_start=start_date)

        macro_df = pd.DataFrame(macro_data)
        macro_df.index = pd.to_datetime(macro_df.index)

        # Merge et application du shift de 30 jours (comme dans ton preprocessing)
        df = df.merge(macro_df, left_index=True, right_index=True, how='left')
        macro_cols = list(macro_map.keys())
        df[macro_cols] = df[macro_cols].ffill().shift(30)

        # Calculs dérivés
        df['Inflation_var'] = df['Inflation_Idx'].pct_change(periods=252)
        df['Yield_Spread'] = df['yield_10y'] - df['yield_2y']
        df['Yield_Spread_Change'] = df['Yield_Spread'].diff()

    except Exception as e:
        st.error(f"Erreur FRED : {e}")

    # 3. Shift Technique Final (shift 1 pour éviter le look-ahead)
    cols_to_fix = ['RSI', 'MFI', 'ADX', 'Volatility', 'Z_Score_10', 'Dist_SMA_10', 'Dist_SMA_50', 'Dist_SMA_200']
    df[cols_to_fix] = df[cols_to_fix].shift(1)

    return df.tail(1)


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Trading ML Dashboard", layout="wide")

st.sidebar.title("Entraînement")
model_type_train = st.sidebar.selectbox("Modèle", ["classification", "regression"])
horizon_train = st.sidebar.slider("Horizon (jours)", 1, 30, 5)

if st.sidebar.button("Lancer l'entrainement"):
    with st.spinner("Exécution Kedro..."):
        params_to_send = {"model_type": model_type_train, "horizon": horizon_train}
        try:
            with KedroSession.create(project_path=project_path, runtime_params=params_to_send) as session:
                session.run()
            st.sidebar.success("Pipeline terminé")
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

st.title("Performance & Live Prediction")

# Récupération MLflow
client = mlflow.tracking.MlflowClient()
try:
    exp = client.get_experiment_by_name("Default")
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
except:
    runs = []

if runs:
    run = runs[0]
    run_id = run.info.run_id
    params = run.data.params

    feature_names = params.get("feature_names", "").split(",")
    saved_model_type = params.get("model_type", "classification")

    st.subheader(f"Dernier modèle : {run.data.tags.get('mlflow.runName')} (Horizon {params.get('horizon')}j)")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe Ratio", f"{run.data.metrics.get('trading_sharpe', 0):.2f}")
    c2.metric("Max Drawdown", f"{run.data.metrics.get('trading_max_drawdown', 0):.2f}")
    c3.metric("Final Return", f"{run.data.metrics.get('trading_final_return', 0):.2f}")

    # Prédiction Live
    st.divider()
    if st.button("Calculer la prédiction Live"):
        with st.spinner("Récupération des données Marché + FRED..."):
            live_row = get_live_features("^GSPC")
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

            # Sélection des features exactes utilisées au training
            X_live = live_row[feature_names]

            if saved_model_type == "classification":
                prob = model.predict_proba(X_live)[0, 1]
                st.metric("Confiance Hausse", f"{prob * 100:.2f} %")
                if prob > 0.5:
                    st.success("SIGNAL : ACHAT")
                else:
                    st.error("SIGNAL : VENTE / NEUTRE")
            else:
                pred = model.predict(X_live)[0] / 1000  # On réajuste le scale *1000 du training
                st.metric("Prédiction (Log Ret)", f"{pred:.5f}")
                st.info(f"Variation attendue : {np.exp(pred) - 1:.4%}")

            # Debug : vérifier que les colonnes ne sont plus à 0
            with st.expander("Voir les features injectées"):
                st.dataframe(X_live)

    # Backtest Plots
    st.divider()
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="test_backtest_curve.html")
        with open(path, 'r') as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    except:
        st.warning("Graphique de backtest non trouvé.")

else:
    st.info("Aucun run MLflow trouvé.")