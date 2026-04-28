from kedro.pipeline import Pipeline, node, pipeline
from .nodes import hyperparameter_optimization, train_best_model, evaluate_and_search_thresholds

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Étape 1
        node(
            func=hyperparameter_optimization,
            inputs=["X_train", "y_train", "X_val", "y_val"],
            outputs="best_model_params",
            name="optimize_model_node",
        ),
        # Étape 2
        node(
            func=train_best_model,
            inputs=["X_train", "y_train", "best_model_params"],
            outputs="model_trained",
            name="train_model_node",
        ),
        # Étape 3
        node(
            func=evaluate_and_search_thresholds,
            inputs=["model_trained", "X_test", "market_logs_test", "params:trading_configs", "params:horizon"],
            outputs="best_trading_params",
            name="grid_search_thresholds_node",
        ),
    ])