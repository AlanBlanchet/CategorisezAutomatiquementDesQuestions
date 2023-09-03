"""
Custom trainer used to log the learning rate
"""

import mlflow
from transformers import Trainer


class SOTrainer(Trainer):
    def log(self, logs: dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        mlflow.log_metric("learning_rate", logs["learning_rate"])
        super().log(logs)
