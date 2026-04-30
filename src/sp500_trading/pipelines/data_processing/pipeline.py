from kedro.pipeline import Pipeline, pipeline, node
from .nodes import dl_data, preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=dl_data,
            inputs=["params:ticker_name", "params:start_date"],
            outputs="market_data_raw",
            name="download_data_node"
        ),
        node(
            func=preprocess_data,
            inputs=["market_data_raw", "params:features", "params:target_num"],
            outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "market_logs_test","market_logs_val"],
            name="preprocess_data_node"
        )
    ])