"""
Module for custom metrics
"""

import numpy as np


def top_k(y_true: np.ndarray, y_preds: np.ndarray, k: int):
    y_preds = y_preds.argsort(axis=-1)[..., ::-1][..., :k]

    correct = 0
    for target, preds in zip(y_true, y_preds):
        if target in preds:
            correct += 1

    return correct / len(y_true)


if __name__ == "__main__":
    true = np.array([1, 2, 3, 4])
    preds = np.array(
        [
            [0.1, 0.5, 0.2, 0.8, 0.9],
            [0.2, 0.9, 0.4, 0.9, 0.9],
            [0.1, 0.1, 0.1, 0.6, 0.08],
            [0.1, 0.05, 0, 0, 0],
        ]
    )

    print(top_k(true, preds, 1))
    print(top_k(true, preds, 2))
    print(top_k(true, preds, 3))
    print(top_k(true, preds, 4))
