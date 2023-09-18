"""
Custom trainer used to log the learning rate
"""

import mlflow
from transformers import Trainer


class SOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.use_mlflow = kwargs.pop("mlflow", True)
        super(SOTrainer, self).__init__(*args, **kwargs)

    def log(self, logs: dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        if self.use_mlflow:
            mlflow.log_metric("learning_rate", logs["learning_rate"])
        super().log(logs)
