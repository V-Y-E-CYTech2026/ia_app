import pandas as pd
import numpy as np
import optuna
import mlflow
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from optuna.samplers import TPESampler


def hyperparameter_optimization(X_train, y_train, X_val, y_val):
    """1. Trouve le meilleur cerveau (XGBoost) via Optuna sur le set de Validation."""

    def objective(trial):
        # On définit les paramètres à tester
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
            # On réduit un peu le max pour gagner du temps
            "max_depth": trial.suggest_int("max_depth", 6, 15),  # ON FORCE LE CERVEAU MUSCLÉ (8 à 15)
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),  # Plus réactif
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            # Plus bas = autorise des patterns plus fins
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0, 0.5),  # Ajoute un peu de piment pour la structure des arbres
        }
        # On gère le weight_ratio séparément pour ne pas faire planter XGBoost
        weight_ratio = trial.suggest_float("weight_ratio", 1, 1.3)

        y_train_vals = y_train.values.ravel()
        weights = np.where(y_train_vals == 0, weight_ratio, 1.0)

        model = XGBClassifier(**param, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train_vals, sample_weight=weights)

        # On évalue sur le set de VALIDATION pour la rigueur
        preds = model.predict(X_val)
        return f1_score(y_val, preds,average='macro')

    # Reproductibilité
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # On lance les 100 trials
    study.optimize(objective, n_trials=100)

    print(f"Meilleurs paramètres trouvés : {study.best_params}")
    return study.best_params


def train_best_model(X_train, y_train, best_params):
    """2. Entraîne le modèle final avec les gagnants d'Optuna."""
    # On extrait le weight_ratio du dictionnaire pour le mettre dans le fit
    params = best_params.copy()
    wr = params.pop("weight_ratio", 1.0)

    y_train_vals = y_train.values.ravel()
    weights = np.where(y_train_vals == 0, wr, 1.0)

    model = XGBClassifier(**params, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train_vals, sample_weight=weights)

    return model


def evaluate_and_search_thresholds(model, X_test, market_logs_test, trading_configs, horizon):
    """3. Teste tous les seuils et trace TOUTES les tentatives pour comparaison."""
    probs = model.predict_proba(X_test)[:, 1]
    mkt_return = market_logs_test.values.ravel()

    best_return = -np.inf
    best_config = {}
    best_df_res = None
    best_signals = None

    # On prépare le graphique dès le début
    fig = go.Figure()

    # On trace le S&P 500 en premier (le benchmark)
    cum_mkt = np.exp(np.cumsum(mkt_return))
    fig.add_trace(go.Scatter(x=X_test.index, y=cum_mkt, name="S&P 500",
                             line=dict(color='white', width=2), opacity=0.8))

    print(f"Probabilité Min: {probs.min():.4f} | Max: {probs.max():.4f} | Moyenne: {probs.mean():.4f}")

    for config in trading_configs:
        signals = np.zeros(len(probs))
        for i in range(0, len(probs), horizon):
            p = probs[i]
            pos = 1 if p > config['seuil_achat'] else (-1 if p < config['seuil_short'] else 0)
            end = min(i + horizon, len(probs))
            signals[i:end] = pos

        strat_ret = signals * mkt_return
        cum_strat = np.exp(np.cumsum(strat_ret))
        final_perf = cum_strat[-1]

        # ON TRACE CHAQUE TEST : Ligne grise très fine et transparente
        fig.add_trace(go.Scatter(
            x=X_test.index, y=cum_strat,
            name=f"A:{config['seuil_achat']} S:{config['seuil_short']}",
            line=dict(color='gray', width=0.5),
            opacity=0.3,
            showlegend=False  # On cache de la légende pour pas que ce soit le bazar
        ))

        with mlflow.start_run(run_name=f"Threshold_{config['seuil_achat']}_{config['seuil_short']}", nested=True):
            mlflow.log_params(config)
            mlflow.log_metric("final_return", final_perf)

        if final_perf > best_return:
            best_return = final_perf
            best_config = config
            best_signals = signals
            best_df_res = pd.DataFrame(index=X_test.index)
            best_df_res['strat_ret'] = strat_ret

    cum_best = np.exp(np.cumsum(best_df_res['strat_ret']))
    fig.add_trace(go.Scatter(
        x=X_test.index, y=cum_best,
        name=f"BEST (A:{best_config['seuil_achat']} S:{best_config['seuil_short']})",
        line=dict(color='gold', width=4)
    ))

    # --- BLOC DEBUG STATS ---
    total_jours = len(best_signals)
    nb_achats = np.sum(best_signals[::horizon] == 1)
    nb_shorts = np.sum(best_signals[::horizon] == -1)
    nb_neutre = np.sum(best_signals[::horizon] == 0)

    changements = 0
    for i in range(horizon, len(best_signals), horizon):
        if best_signals[i] != best_signals[i - horizon]:
            changements += 1

    print(f"\n--- DEBUG STATS MEILLEUR MODÈLE ({best_config['seuil_achat']}) ---")
    print(f"Nombre total de jours : {total_jours}")
    print(f"ACHATS: {nb_achats} | SHORTS: {nb_shorts} | NEUTRES: {nb_neutre}")
    print(f"CHANGEMENTS de position : {changements}\n")

    fig.update_layout(
        title=f"Comparaison des Seuils - Meilleur: {best_return:.2f}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Performance (Base 1.0)"
    )

    mlflow.log_figure(fig, "all_thresholds_comparison.html")
    return best_config