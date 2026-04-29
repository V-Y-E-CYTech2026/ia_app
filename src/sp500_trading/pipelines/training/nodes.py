import pandas as pd
import numpy as np
import optuna
import mlflow
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from optuna.samplers import TPESampler

from sklearn.metrics import matthews_corrcoef


def hyperparameter_optimization(X_train, y_train, X_val, y_val):
    """1. Trouve le meilleur cerveau (XGBoost) via Optuna en maximisant le MCC."""

    def objective(trial):
        # Paramètres optimisés pour la vitesse et la robustesse
        param = {
            # 1. On garde le nombre d'estimateurs, mais on ajoute un early stopping au besoin
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),

            # 2. Profondeur plus faible pour la généralisation
            "max_depth": trial.suggest_int("max_depth", 3, 6),

            # 3. Apprentissage lent = meilleur modèle
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),

            # 4. Régularisation L1 et L2 (Crucial !)
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

            # 5. Plus conservateur sur le poids des enfants
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),

            # 6. Échantillonnage pour ajouter du hasard (Bruit de combat)
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),

            # 7. Gamma (plus il est haut, plus le modèle est conservateur)
            "gamma": trial.suggest_float("gamma", 0.1, 1.0),
        }

        # On fixe le weight_ratio à 1 pour laisser le MCC équilibrer naturellement
        y_train_vals = y_train.values.ravel()

        model = XGBClassifier(**param, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train_vals)

        # Prédiction sur le set de VALIDATION
        preds = model.predict(X_val)

        # On utilise le MCC au lieu du F1 Macro
        score = matthews_corrcoef(y_val, preds)

        return score

    # Configuration de l'étude Optuna
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # On lance les 100 trials
    study.optimize(objective, n_trials=100)

    print(f"--- OPTIMISATION TERMINÉE ---")
    print(f"Meilleur MCC trouvé : {study.best_value:.4f}")
    print(f"Meilleurs paramètres : {study.best_params}")

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


def evaluate_and_search_thresholds(model, X_test, market_logs_test, horizon):
    """3. Teste indépendamment les seuils d'achat et de short autour de la moyenne."""
    probs = model.predict_proba(X_test)[:, 1]
    mkt_return = market_logs_test.values.ravel()
    mean_p = probs.mean()

    # On définit des listes d'offsets (tu peux les différencier si besoin)
    test_offsets_achat = [0,0.01, 0.05,0.06, 0.07, 0.10, 0.15, 0.20]
    test_offsets_short = [0.01, 0.05, 0.07, 0.10, 0.15,0.17, 0.20,0.22,0.25,0.3,0.4]

    best_return = -np.inf
    best_config = {}
    best_df_res = None
    best_signals = None

    fig = go.Figure()
    cum_mkt = np.exp(np.cumsum(mkt_return))
    fig.add_trace(go.Scatter(x=X_test.index, y=cum_mkt, name="S&P 500",
                             line=dict(color='white', width=2), opacity=0.8))

    print(f"Probabilité Moyenne: {mean_p:.4f} (Exploration de la grille Achat/Short)")

    # Double boucle pour tester l'indépendance des deux côtés
    for off_a in test_offsets_achat:
        for off_s in test_offsets_short:
            # On cast en float tout de suite pour le JSON final
            config = {
                'seuil_achat': float(round(mean_p + off_a, 4)),
                'seuil_short': float(round(mean_p - off_s, 4))
            }

            signals = np.zeros(len(probs))
            for i in range(0, len(probs), horizon):
                p = probs[i]
                pos = 1 if p > config['seuil_achat'] else (-1 if p < config['seuil_short'] else 0)
                end = min(i + horizon, len(probs))
                signals[i:end] = pos

            strat_ret = signals * mkt_return
            cum_strat = np.exp(np.cumsum(strat_ret))
            final_perf = cum_strat[-1]

            # On trace les tentatives (25 au total ici)
            fig.add_trace(go.Scatter(
                x=X_test.index, y=cum_strat,
                name=f"A:{config['seuil_achat']} S:{config['seuil_short']}",
                line=dict(color='gray', width=0.5),
                opacity=0.2,  # Plus transparent car il y a plus de lignes
                showlegend=False
            ))

            if final_perf > best_return:
                best_return = final_perf
                best_config = config
                best_signals = signals
                best_df_res = pd.DataFrame(index=X_test.index, data={'strat_ret': strat_ret})

    # Plot de la meilleure courbe
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

    print(f"\n--- DEBUG STATS MEILLEUR MODÈLE ---")
    print(f"Seuils Optimaux -> Achat: {best_config['seuil_achat']} | Short: {best_config['seuil_short']}")
    print(f"ACHATS: {nb_achats} | SHORTS: {nb_shorts} | NEUTRES: {nb_neutre}")
    print(f"CHANGEMENTS de position : {changements}")
    print(f"Rendement final : {best_return:.2f}\n")

    fig.update_layout(
        title=f"Grille de Seuils Indépendants - Meilleur: {best_return:.2f}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Performance (Base 1.0)"
    )

    mlflow.log_figure(fig, "all_thresholds_comparison.html")
    print("--deugé",best_config)
    # Pas besoin de refaire le dictionnaire, ils sont déjà en float
    return best_config