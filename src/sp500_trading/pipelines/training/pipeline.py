from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    hyperparameter_optimization,
    train_best_model,
    evaluate_and_search_thresholds,
    plot_feature_importance,
    plot_model_calibration,
    log_model_metrics
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # 1. Optimisation des hyperparamètres avec Optuna
            node(
                func=hyperparameter_optimization,
                inputs=["X_train", "y_train", "X_val", "y_val"],
                outputs="best_hyperparams",
                name="hyperparameter_optimization_node",
            ),

            # 2. Entraînement du modèle final avec les meilleurs paramètres
            node(
                func=train_best_model,
                inputs=["X_train", "y_train", "best_hyperparams"],
                outputs="best_model",
                name="train_best_model_node",
            ),

            # 3. Recherche des seuils optimaux (Achat/Short) et Backtest
            node(
                func=evaluate_and_search_thresholds,
                inputs=["best_model", "X_test", "market_logs_test", "params:horizon"],
                outputs="best_thresholds",
                name="evaluate_thresholds_node",
            ),


            node(
                func=plot_feature_importance,
                inputs=["best_model", "X_train"],
                outputs="feature_importance_report",
                name="feature_importance_node",
            ),

            # 5. Courbe de calibration (Fiabilité des probabilités)
            node(
                func=plot_model_calibration,
                inputs=["best_model", "X_test", "y_test"],
                outputs=None,
                name="model_calibration_node",
            ),
            node(
                func=log_model_metrics,
                inputs=[
                    "best_model",
                    "X_test",
                    "y_test",
                    "best_thresholds",
                    "market_logs_test",
                    "params:horizon"
                ],
                outputs=None,
                name="log_metrics_node",
            ),
        ]
    )