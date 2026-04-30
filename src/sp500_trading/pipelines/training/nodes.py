import pandas as pd
import numpy as np
import optuna
import mlflow
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def hyperparameter_optimization(X_train, y_train, horizon):
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "gamma": trial.suggest_float("gamma", 0, 0.7),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.80, 1.0)
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_v = y_train.iloc[train_idx].values.ravel(), y_train.iloc[val_idx].values.ravel()

            model = XGBClassifier(**param, early_stopping_rounds=10, n_jobs=-1, random_state=42)
            model.fit(X_tr.iloc[::horizon], y_tr[::horizon], eval_set=[(X_v, y_v)], verbose=False)

            preds = model.predict(X_v)
            scores.append(matthews_corrcoef(y_v, preds))

        return np.mean(scores)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=200)

    print(f"Meilleur MCC trouve : {study.best_value:.4f}")
    print(f"Meilleurs parametres : {study.best_params}")

    return study.best_params


def train_best_model(X_train, y_train, best_params, horizon):
    params = best_params.copy()
    params.pop("weight_ratio", None)

    base_model = XGBClassifier(**params, n_jobs=-1, random_state=42)

    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)

    calibrated_model.fit(X_train.iloc[::horizon], y_train.values.ravel()[::horizon])

    return calibrated_model


def evaluate_and_search_thresholds(model, X_val, market_logs_val, horizon):
    probs = model.predict_proba(X_val)[:, 1]
    mkt_return = market_logs_val.values.ravel()
    mean_p = probs.mean()

    print(f"Probabilite Min: {probs.min() * 100:.2f}")
    print(f"Probabilite Max: {probs.max() * 100:.2f}")
    print(f"Probabilite Moyenne: {mean_p:.4f}")

    test_offsets_achat = [-0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.10,0.15]

    test_offsets_short = [0.02, 0.05, 0.08]
    # test_offsets_short = [0.05, 0.10,0.11,0.12,0.13, 0.15, 0.20, 0.25]

    best_return = -np.inf
    best_config = {}
    best_df_res = None
    best_signals = None

    fig = go.Figure()
    cum_mkt = np.exp(np.cumsum(mkt_return))
    fig.add_trace(go.Scatter(x=X_val.index, y=cum_mkt, name="S&P 500", line=dict(color='white', width=2), opacity=0.8))

    for off_a in test_offsets_achat:
        for off_s in test_offsets_short:
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

            fig.add_trace(go.Scatter(
                x=X_val.index, y=cum_strat,
                name=f"A:{config['seuil_achat']} S:{config['seuil_short']}",
                line=dict(color='gray', width=0.5),
                opacity=0.2,
                showlegend=False
            ))

            if final_perf > best_return:
                best_return = final_perf
                best_config = config
                best_signals = signals
                best_df_res = pd.DataFrame(index=X_val.index, data={'strat_ret': strat_ret})

    cum_best = np.exp(np.cumsum(best_df_res['strat_ret']))
    fig.add_trace(go.Scatter(
        x=X_val.index, y=cum_best,
        name=f"BEST (A:{best_config['seuil_achat']} S:{best_config['seuil_short']})",
        line=dict(color='gold', width=4)
    ))

    nb_achats = np.sum(best_signals[::horizon] == 1)
    nb_shorts = np.sum(best_signals[::horizon] == -1)
    nb_neutre = np.sum(best_signals[::horizon] == 0)

    changements = 0
    for i in range(horizon, len(best_signals), horizon):
        if best_signals[i] != best_signals[i - horizon]:
            changements += 1

    print(f"Seuils Optimaux -> Achat: {best_config['seuil_achat']} | Short: {best_config['seuil_short']}")
    print(f"ACHATS: {nb_achats} | SHORTS: {nb_shorts} | NEUTRES: {nb_neutre}")
    print(f"CHANGEMENTS de position : {changements}")
    print(f"Rendement final : {best_return:.2f}")

    fig.update_layout(
        title=f"Grille de Seuils Independants - Meilleur: {best_return:.2f}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Performance (Base 1.0)"
    )

    mlflow.log_figure(fig, "all_thresholds_comparison.html")
    return best_config


def plot_feature_importance(model, X_train):
    if hasattr(model, "calibrated_classifiers_"):
        importances = np.mean(
            [clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_],
            axis=0
        )
    else:
        try:
            importances = model.feature_importances_
        except AttributeError:
            importances = model.estimator.feature_importances_

    df_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    fig = go.Figure(go.Bar(
        x=df_imp["importance"],
        y=df_imp["feature"],
        orientation='h',
        marker_color='gold'
    ))
    fig.update_layout(
        title="Top 20 Features",
        template="plotly_dark",
        yaxis_autorange="reversed"
    )

    mlflow.log_figure(fig, "feature_importance.html")
    return df_imp


def plot_model_calibration(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    fop, mpv = calibration_curve(y_test, probs, n_bins=10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Parfait', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=mpv, y=fop, mode='markers+lines', name='Modele'))

    fig.update_layout(title="Courbe de Calibration", template="plotly_dark")
    mlflow.log_figure(fig, "calibration_curve.html")


def log_model_metrics(model, X_test, y_test, best_thresholds, market_logs_test, horizon):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > best_thresholds['seuil_achat']).astype(int)

    mcc = matthews_corrcoef(y_test, preds)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("model_mcc", mcc)
    mlflow.log_metric("model_accuracy", acc)

    mkt_return = market_logs_test.values.ravel()
    signals = np.zeros(len(probs))

    for i in range(0, len(probs), horizon):
        p = probs[i]
        pos = 1 if p > best_thresholds['seuil_achat'] else (-1 if p < best_thresholds['seuil_short'] else 0)
        end = min(i + horizon, len(probs))
        signals[i:end] = pos

    strat_ret = signals * mkt_return
    cum_strat = np.exp(np.cumsum(strat_ret))
    cum_mkt = np.exp(np.cumsum(mkt_return))

    vol = strat_ret.std() * np.sqrt(252)
    if vol > 0:
        sharpe = (strat_ret.mean() * 252) / vol
    else:
        sharpe = 0

    peak = np.maximum.accumulate(cum_strat)
    drawdown = (cum_strat - peak) / peak
    max_dd = drawdown.min()

    mlflow.log_metric("trading_sharpe", float(sharpe))
    mlflow.log_metric("trading_max_drawdown", float(max_dd))
    mlflow.log_metric("trading_final_return", float(cum_strat[-1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_test.index, y=cum_mkt,
        name="S&P 500 (Test)",
        line=dict(color='white', width=2), opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=X_test.index, y=cum_strat,
        name=f"Strategie (A:{best_thresholds['seuil_achat']} S:{best_thresholds['seuil_short']})",
        line=dict(color='gold', width=3)
    ))

    fig.update_layout(
        title=f"Performance sur le set de TEST - Sharpe: {sharpe:.2f}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Performance (Base 1.0)"
    )

    mlflow.log_figure(fig, "test_backtest_curve.html")

    print(f"MCC: {mcc:.4f}, Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.2f}")