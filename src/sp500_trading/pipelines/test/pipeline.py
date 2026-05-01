from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_feature_importance,
    plot_model_calibration,
    evaluate_test_metrics
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=plot_feature_importance,
                inputs=["best_model", "X_train", "params:model_type"],
                outputs="feature_importance_report",
                name="feature_importance_node",
            ),
            node(
                func=plot_model_calibration,
                inputs=["best_model", "X_test", "y_test", "params:model_type"],
                outputs=None,
                name="model_calibration_node",
            ),
            node(
                func=evaluate_test_metrics,
                inputs=[
                    "best_model",
                    "X_test",
                    "y_test",
                    "best_thresholds",
                    "market_logs_test",
                    "params:horizon",
                    "params:model_type"
                ],
                outputs=None,
                name="log_metrics_node",
            ),
        ]
    )