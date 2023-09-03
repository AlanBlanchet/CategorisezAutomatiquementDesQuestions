"""
Create a callback with the hugging face API
Used to retrieve the loggable param metrics
"""

import mlflow
from transformers import TrainerCallback


def get_callback(mlflow_end):
    class SOCallback(TrainerCallback):
        def __init__(self, **args) -> None:
            super().__init__(**args)

        def on_log(self, args, state, control, logs, model=None, **kwargs):
            if state.is_world_process_zero:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v, step=state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            mlflow_end()

    return SOCallback
