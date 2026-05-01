import pandas as pd
import numpy as np
import mlflow
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve


def plot_feature_importance(model, X_train, model_type="classification"):
    mlflow.set_tag("mlflow.runName", f"importance_variables_{model_type}")
    mlflow.set_tag("pipeline", "model_evaluation")
    mlflow.set_tag("etape", "feature_importance")

    if model_type == "classification" and hasattr(model, "calibrated_classifiers_"):
        importances = np.mean([clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_], axis=0)
    else:
        try:
            importances = model.feature_importances_
        except AttributeError:
            importances = model.estimator.feature_importances_

    df_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    fig = go.Figure(go.Bar(x=df_imp["importance"], y=df_imp["feature"], orientation='h', marker_color='gold'))
    fig.update_layout(title="Top 20 Features", template="plotly_dark", yaxis_autorange="reversed")
    mlflow.log_figure(fig, "feature_importance.html")

    return df_imp


def plot_model_calibration(model, X_test, y_test, model_type="classification"):
    mlflow.set_tag("mlflow.runName", f"calibration_{model_type}")
    mlflow.set_tag("pipeline", "model_evaluation")
    mlflow.set_tag("etape", "courbe_calibration")

    if model_type == "classification":
        probs = model.predict_proba(X_test)[:, 1]
        fop, mpv = calibration_curve(y_test, probs, n_bins=10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Parfait', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=mpv, y=fop, mode='markers+lines', name='Modele'))

        fig.update_layout(title="Courbe de Calibration", template="plotly_dark")
        mlflow.log_figure(fig, "calibration_curve.html")


def evaluate_test_metrics(model, X_test, y_test, best_thresholds, market_logs_test, horizon,
                          model_type="classification"):
    mlflow.set_tag("mlflow.runName", f"evaluation_test_{model_type}")
    mlflow.set_tag("pipeline", "model_evaluation")
    mlflow.set_tag("etape", "metriques_test")

    if model_type == "classification":
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > best_thresholds['seuil_achat']).astype(int)

        # Métriques ML de base
        mlflow.log_metric("model_mcc", matthews_corrcoef(y_test, preds))
        mlflow.log_metric("model_accuracy", accuracy_score(y_test, preds))

        # Nouvelles métriques ML
        mlflow.log_metric("model_precision", precision_score(y_test, preds, zero_division=0))
        mlflow.log_metric("model_recall", recall_score(y_test, preds, zero_division=0))
        mlflow.log_metric("model_f1_score", f1_score(y_test, preds, zero_division=0))
    else:
        probs = model.predict(X_test) / 1000

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

    # --- Métriques de risque classiques ---
    vol = strat_ret.std() * np.sqrt(252)
    if vol > 0:
        sharpe = (strat_ret.mean() * 252) / vol
    else:
        sharpe = 0

    peak = np.maximum.accumulate(cum_strat)
    drawdown = (cum_strat - peak) / peak
    max_dd = drawdown.min()

    # --- Nouvelles métriques de Trading ---

    # 1. Ratio de Sortino (Volatilité à la baisse uniquement)
    downside_returns = strat_ret[strat_ret < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (strat_ret.mean() * 252) / downside_vol if downside_vol > 0 else 0

    # 2. Ratio de Calmar (Rendement annuel / Max Drawdown)
    n_days = len(strat_ret)
    annualized_return = (cum_strat[-1]) ** (252 / n_days) - 1
    calmar = annualized_return / abs(max_dd) if max_dd < 0 else 0

    # 3. Win Rate (Taux de réussite des jours où on est investi)
    active_days = strat_ret[signals != 0]
    win_rate = len(active_days[active_days > 0]) / len(active_days) if len(active_days) > 0 else 0

    # Logs MLflow pour le trading
    mlflow.log_metric("trading_sharpe", float(sharpe))
    mlflow.log_metric("trading_max_drawdown", float(max_dd))
    mlflow.log_metric("trading_final_return", float(cum_strat[-1]))
    mlflow.log_metric("trading_sortino", float(sortino))
    mlflow.log_metric("trading_calmar", float(calmar))
    mlflow.log_metric("trading_win_rate", float(win_rate))

    # Graphique mis à jour avec le Sortino dans le titre
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=X_test.index, y=cum_mkt, name="S&P 500 (Test)", line=dict(color='white', width=2), opacity=0.8))
    fig.add_trace(go.Scatter(x=X_test.index, y=cum_strat,
                             name=f"Strategie (A:{best_thresholds['seuil_achat']} S:{best_thresholds['seuil_short']})",
                             line=dict(color='gold', width=3)))

    fig.update_layout(title=f"Performance TEST - Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}", template="plotly_dark",
                      xaxis_title="Date",
                      yaxis_title="Performance")
    mlflow.log_figure(fig, "test_backtest_curve.html")