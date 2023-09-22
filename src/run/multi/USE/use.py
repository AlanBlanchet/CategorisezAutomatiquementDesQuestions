"""
Train a bert model on our dataset
"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import yaml
from gradient_accumulator import GradientAccumulateModel
from keras import backend as K
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
from tensorflow_hub import KerasLayer

from src.dataset.dataset import Dataset

os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"


if __name__ == "__main__":
    embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    output_p = (Path(__file__).parents[4] / "output").resolve()
    output_p.mkdir(exist_ok=True)

    use_p = output_p / "USE"
    use_p.mkdir(exist_ok=True)

    r = 1
    use_run_p = use_p / f"run{r}"
    while use_run_p.exists():
        r += 1
        use_run_p = use_p / f"run{r}"

    use_run_p.mkdir()

    n = 50_000
    target_min_freq = 0.008
    epochs = 300
    batch_size = 64 * 4
    gradient_accumulation_steps = 8
    lr = 1e-3
    lr_decay = 0.99
    lr_decay_steps = 40

    with open(str(use_run_p / "params.txt"), "w+") as f:
        f.write(
            f"{n=}\n{epochs=}\n{target_min_freq=}\n{batch_size=}\n{lr=}\n{lr_decay=}\n{lr_decay_steps=}\n"
        )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=lr_decay_steps, decay_rate=lr_decay, staircase=False
    )

    topics = Dataset("topics1", n=n)

    all_targets = [f"target{t}" for t in range(1, 6)]

    # We need at least more than one of a class
    more_than_one = pd.Series(
        topics.df[all_targets].values.flatten()
    ).value_counts() > int(n * target_min_freq)
    invalid_targets = more_than_one[more_than_one == False].index

    # Line contains targets that appear more than once !
    # Removing targets from the 5 targets array
    multi_targets = topics.df[all_targets].parallel_apply(
        lambda x: list(filter(lambda y: y not in invalid_targets, x)),
        axis=1,
    )

    not_empty = ~multi_targets.parallel_apply(lambda x: len(x)).isin([0])
    df = topics.df.loc[not_empty]
    multi_targets = multi_targets[not_empty]

    # Weight distribution
    nb_pos = multi_targets.explode().value_counts()
    weights = nb_pos / nb_pos.sum()
    weights = nb_pos

    # Here we make the binarizer learn our targets in advance to save memory
    mlb = MultiLabelBinarizer()
    ohe_targets = mlb.fit_transform(multi_targets)

    num_classes = len(mlb.classes_)
    print(f"{num_classes=}")

    df["text"] = df["original_title"] + df["original_text"]

    df["labels"] = df.parallel_apply(
        lambda row: np.array(
            mlb.transform([multi_targets[row.name]])[0], dtype=np.float32
        ),
        axis=1,
    )

    # INFO
    # Bug with nested numpy arrays
    converted = np.array([l.tolist() for l in df["labels"].values], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        converted,
        random_state=0,
    )

    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    print("Example :")
    print(X_train.dtype, X_train[0])
    print("Targets :")
    print(y_train.dtype, y_train[0])
    print(mlb.inverse_transform(np.array([y_train[0]]))[0])

    print("Creating trainer")

    id2label = {i: cls for i, cls in enumerate(mlb.classes_)}
    label2id = {v: k for k, v in id2label.items()}

    # Save run data
    with open(str(use_run_p / "id2label.yaml"), "w") as f:
        yaml.dump(id2label, f)

    with open(str(use_run_p / "label2id.yaml"), "w") as f:
        yaml.dump(label2id, f)

    with open(str(use_run_p / "classes.yaml"), "w") as f:
        yaml.dump(mlb.classes_, f)

    in_layer = Input(shape=[], dtype=tf.string)
    embedding_layer = KerasLayer(
        embedding,
        trainable=False,
    )(in_layer)
    d1_layer = Dense(
        256, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(embedding_layer)
    dropout = Dropout(0.2)(d1_layer)
    classifier = Dense(num_classes, activation="sigmoid")(dropout)

    model = Model(in_layer, classifier)
    # model = GradientAccumulateModel(
    #     accum_steps=gradient_accumulation_steps,
    #     inputs=model.input,
    #     outputs=model.output,
    # )

    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def lr_metric(y_true, y_pred):
        return opt.lr

    def jaccard_samples(y_true, y_pred_sig):
        y_pred = np.zeros(y_pred_sig.shape)
        y_pred[np.where(y_pred_sig.numpy() >= 0.5)] = 1
        return jaccard_score(
            y_true.numpy(), y_pred, average="samples", zero_division=0.0
        )

    def jaccard_micro(y_true, y_pred_sig):
        y_pred = np.zeros(y_pred_sig.shape)
        y_pred[np.where(y_pred_sig.numpy() >= 0.5)] = 1
        return jaccard_score(y_true.numpy(), y_pred, average="micro", zero_division=0.0)

    def jaccard_macro(y_true, y_pred_sig):
        y_pred = np.zeros(y_pred_sig.shape)
        y_pred[np.where(y_pred_sig.numpy() >= 0.5)] = 1
        return jaccard_score(y_true.numpy(), y_pred, average="macro", zero_division=0.0)

    weights.index = [label2id[i] for i in weights.index]
    weights = weights.sort_index()
    # w = np.zeros(len(weights))
    # for id, label in id2label.items():
    #     w[id] = weights[label]

    # w = tf.Variable(w)
    # w = tf.cast(w, tf.float32)
    print("Weights :")
    print(weights.to_dict())

    # def weighted_binary_crossentropy(y_true, y_pred):
    #     y_true = tf.cast(y_true, y_pred.dtype)
    #     return tf.math.reduce_mean(
    #         tf.keras.losses.binary_focal_crossentropy(y_true, y_pred)
    #         * tf.math.reduce_sum(w * y_true, axis=-1)
    #     )

    # print("Weights :")
    # print(weights)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        # loss=tf.compat.v1.losses.sigmoid_cross_entropy(),
        # loss=lambda y_true, y_pred: tf.nn.sigmoid_cross_entropy_with_logits(
        #     y_true, tf.sigmoid(y_pred)
        # ),
        optimizer=opt,
        metrics=[
            tf.keras.metrics.F1Score(average="macro"),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            lr_metric,
            jaccard_samples,
            jaccard_micro,
            jaccard_macro,
        ],
        run_eagerly=True,
    )

    end_run = topics.mlflow_run(use_run_p.stem)
    del topics
    del multi_targets

    class MetricCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs != None:
                for name, value in logs.items():
                    mlflow.log_metric(name, value, epoch)

        def on_train_end(self, logs=None):
            end_run()

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[
            MetricCallback(),
        ],
        class_weight=weights.to_dict(),
    )

    model.save(str(use_run_p / "model"))

    def compute_metrics(y_true, y_pred_sig):
        threshold = 0.5

        y_pred = np.zeros(y_pred_sig.shape)
        y_pred[np.where(y_pred_sig >= threshold)] = 1

        y_labels = mlb.inverse_transform(y_pred)
        labels = mlb.inverse_transform(y_true)

        pd.DataFrame(y_pred_sig).to_csv(use_run_p / "y_pred_matrix.csv".format())

        pd.DataFrame({"target": labels, "pred": y_labels}).to_csv(
            use_run_p / "example.csv".format()
        )

        avg_metrics = {
            "jaccard_score": {"f": jaccard_score},
            "f1_score": {"f": f1_score},
            "roc_auc_score": {"f": roc_auc_score},
            "precision_score": {"f": precision_score, "zero_division": 0.0},
            "recall_score": {"f": recall_score, "zero_division": 0.0},
        }
        avg_types = ["micro", "macro", "weighted", "samples"]

        metrics = {
            **{
                f"{name}_{avg}": score["f"](
                    y_true,
                    y_pred,
                    average=avg,
                    # Add custom args
                    **{sk: sv for sk, sv in score.items() if sk != "f"},
                )
                for avg in avg_types
                for name, score in avg_metrics.items()
            },
            "accuracy_score": accuracy_score(y_true, y_pred),
        }

        return metrics

    y_pred = model.predict(X_test, batch_size=batch_size)

    metrics = compute_metrics(y_test.numpy(), y_pred)

    metrics_text = ""
    for name, val in metrics.items():
        metrics_text += f"{name} = {val}\n"

    with open(use_run_p / "metrics.txt", "w+") as f:
        f.write(metrics_text)

    print(y_pred.shape)
