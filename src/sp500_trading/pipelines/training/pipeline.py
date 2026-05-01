from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    hyperparameter_optimization,
    train_best_model,
    evaluate_and_search_thresholds
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=hyperparameter_optimization,
                inputs=["X_train", "y_train", "params:horizon", "params:model_type"],
                outputs="best_hyperparams",
                name="hyperparameter_optimization_node",
            ),
            node(
                func=train_best_model,
                inputs=["X_train", "y_train", "best_hyperparams", "params:horizon", "params:model_type"],
                outputs="best_model",
                name="train_best_model_node",
            ),
            node(
                func=evaluate_and_search_thresholds,
                inputs=["best_model", "X_val", "market_logs_val", "params:horizon", "params:model_type"],
                outputs="best_thresholds",
                name="evaluate_thresholds_node",
            ),
        ]
    )