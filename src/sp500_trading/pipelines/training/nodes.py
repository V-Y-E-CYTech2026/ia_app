import pandas as pd
import numpy as np
import optuna
import mlflow
import plotly.graph_objects as go
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import matthews_corrcoef, mean_squared_error, mean_absolute_error
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from mlflow.models.signature import infer_signature

def hyperparameter_optimization(X_train, y_train, horizon, model_type="classification"):
    mlflow.set_tag("mlflow.runName", f"optimisation_{model_type}")
    mlflow.set_tag("pipeline", "model_training")
    mlflow.set_tag("etape", "hyperparametres")
    mlflow.log_param("horizon", horizon)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("feature_names", ",".join(X_train.columns.tolist()))

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        if model_type == "classification":
            param = {"n_estimators": trial.suggest_int("n_estimators", 20, 600),
                     "max_depth": trial.suggest_int("max_depth", 2, 8),
                     "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
                     "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                     "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                     "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                     "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.95, 1.05)}
        else :
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 1500),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
                "gamma": trial.suggest_float("gamma", 0, 0.7)
            }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_v = y_train.iloc[train_idx].values.ravel(), y_train.iloc[val_idx].values.ravel()

            if model_type == "classification":
                model = XGBClassifier(**param, n_jobs=-1, random_state=42)
                model.fit(X_tr.iloc[::horizon], y_tr[::horizon], eval_set=[(X_v, y_v)], verbose=False)
                preds = model.predict(X_v)
                scores.append(matthews_corrcoef(y_v, preds))
            else:
                model = XGBRegressor(**param, n_jobs=-1, random_state=42)
                model.fit(X_tr.iloc[::horizon], y_tr[::horizon], eval_set=[(X_v, y_v)], verbose=False)
                preds = model.predict(X_v)
                scores.append(-mean_squared_error(y_v, preds))

        return np.mean(scores)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=200)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_score", study.best_value)

    return study.best_params

def train_best_model(X_train, y_train, best_params, horizon, model_type="classification"):
    mlflow.set_tag("mlflow.runName", f"entrainement_{model_type}")
    mlflow.set_tag("pipeline", "model_training")
    mlflow.set_tag("etape", "entrainement_final")

    params = best_params.copy()
    if model_type == "classification":
        base_model = XGBClassifier(**params, n_jobs=-1, random_state=42)
        final_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    else:
        final_model = XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1, random_state=42)

    final_model.fit(X_train.iloc[::horizon], y_train.values.ravel()[::horizon])

    sample_input = X_train.head(3)
    sample_output = final_model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model", signature=signature)
    mlflow.log_param("feature_names", ",".join(X_train.columns.tolist()))
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("horizon", horizon)

    return final_model

def evaluate_and_search_thresholds(model, X_val, market_logs_val, horizon, model_type="classification"):
    mlflow.set_tag("mlflow.runName", f"evaluation_{model_type}")
    mlflow.set_tag("pipeline", "model_training")
    mlflow.set_tag("etape", "recherche_seuils")

    if model_type == "classification":
        probs = model.predict_proba(X_val)[:, 1]
        test_offsets_achat = [-0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.10, 0.15]
        test_offsets_short = [0.05, 0.10, 0.11, 0.12, 0.13, 0.15, 0.20, 0.25]
        mean_p = probs.mean()
    else:
        probs = model.predict(X_val)/1000
        test_offsets_achat = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
        test_offsets_short = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
        mean_p = 0.0

    mkt_return = market_logs_val.values.ravel()
    best_return = -np.inf
    best_sharpe = -np.inf
    best_config = {}
    best_df_res = None

    fig = go.Figure()
    cum_mkt = np.exp(np.cumsum(mkt_return))
    fig.add_trace(go.Scatter(x=X_val.index, y=cum_mkt, name="S&P 500", line=dict(color='white', width=2), opacity=0.8))

    for off_a in test_offsets_achat:
        for off_s in test_offsets_short:
            if model_type == "classification":
                config = {
                    'seuil_achat': float(round(mean_p + off_a, 4)),
                    'seuil_short': float(round(mean_p - off_s, 4))
                }
            else:
                config = {
                    'seuil_achat': float(off_a),
                    'seuil_short': float(-off_s)
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

            ann_ret = strat_ret.mean() * 252
            ann_vol = strat_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else -1e9

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_return = final_perf
                best_config = config
                best_df_res = pd.DataFrame(index=X_val.index, data={'strat_ret': strat_ret})

            fig.add_trace(go.Scatter(x=X_val.index, y=cum_strat, line=dict(color='gray', width=0.5), opacity=0.2, showlegend=False))

    mlflow.log_params(best_config)
    mlflow.log_metric("rendement_strategie_max", best_return)
    mlflow.log_metric("rendement_sp500", cum_mkt[-1])
    mlflow.log_metric("best_val_sharpe", best_sharpe)

    cum_best = np.exp(np.cumsum(best_df_res['strat_ret']))
    fig.add_trace(go.Scatter(x=X_val.index, y=cum_best, line=dict(color='gold', width=4), name="BEST CONFIG"))
    fig.update_layout(title=f"Optimisation Seuils - Sharpe: {best_sharpe:.2f}", template="plotly_dark")
    mlflow.log_figure(fig, "val_thresholds_comparison.html")

    return best_config