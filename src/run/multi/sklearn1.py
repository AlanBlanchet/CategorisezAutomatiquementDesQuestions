"""
This is the module used to compare the best params of the models
to find the best model to use
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import dump
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from src.dataset.dataset import Dataset

pandarallel.initialize(verbose=0)


def train(model: OneVsRestClassifier, X_train, X_test, y_train, y_true, output_p: Path):
    name = type(model.estimator).__name__

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    calculate_metrics(y_true, y_pred, output_p, name, train_time, predict_time)

    return model


def calculate_metrics(
    y_true, y_pred, output_p, name="Model", train_time=0, predict_time=0
):
    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=1)
    recall = recall_score(y_true, y_pred, average="samples", zero_division=1)
    precision = precision_score(y_true, y_pred, average="samples", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="samples", zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    res = f"{name:<24} {jaccard=:<10.3f} {accuracy=:<10.3f} {precision=:<10.3f} {recall=:<10.3f} {f1=:<10.3f} {train_time=:<10.3f}s {predict_time=:<10.3f}s"
    with open(output_p / "result.txt", "a+") as f:
        f.write(res + "\n")
    print(res)


def output(mlb: MultiLabelBinarizer, model, X_test, y_test, save_p: Path):
    inverse_test = mlb.inverse_transform(y_test)
    inverse = mlb.inverse_transform(model.predict(X_test))

    pd.DataFrame(
        dict(
            [(i, pd.Series([inverse_test[i], inverse[i]])) for i in range(len(y_test))]
        )
    ).T.to_csv(save_p)


if __name__ == "__main__":
    with tqdm(total=5, leave=False, position=0) as pbar:
        update = lambda x, y=1: (pbar.update(y), pbar.set_description(x))

        output_p = (Path(__file__).parents[3] / "output").resolve()
        output_p.mkdir(exist_ok=True)

        sklearn_p = output_p / "sklearn"
        sklearn_p.mkdir(exist_ok=True)

        r = 1
        loc_p = sklearn_p / f"run{r}"
        while loc_p.exists():
            r += 1
            loc_p = sklearn_p / f"run{r}"

        loc_p.mkdir(exist_ok=True)

        print("Output to ", loc_p)

        update("Import dataset", 0)

        n = 50000
        target_min_freq = 0.005

        topics = Dataset("topics1", n=n, sort=False)

        update("Counting invalid targets")

        all_targets = [f"target{t}" for t in range(1, 6)]

        # We need at least more than one of a class
        more_than_one = pd.Series(
            topics.df[all_targets].values.flatten()
        ).value_counts() > int(n * target_min_freq)
        invalid_targets = more_than_one[more_than_one == False].index
        # Line contains targets that appear more than once !
        update("Removing invalid targets")
        # Removing targets from the 5 targets array
        multi_targets = topics.df[all_targets].parallel_apply(
            lambda x: list(filter(lambda y: y not in invalid_targets, x)),
            axis=1,
        )

        not_empty = ~multi_targets.parallel_apply(lambda x: len(x)).isin([0])
        df = topics.df.loc[not_empty]
        multi_targets = multi_targets[not_empty]

        update("Binarizing targets")
        # Here we make the binarizer learn our targets in advance to save memory
        mlb = MultiLabelBinarizer()
        ohe_targets = mlb.fit_transform(multi_targets)

        text = df["title"] + df["text"]

        update("Vectorizing")
        vectorizer = TfidfVectorizer(min_df=0.00001, max_df=0.99)
        ohe = vectorizer.fit_transform(text)
        num_classes = len(mlb.classes_)

        print(f"Dataset {num_classes=} vectors={ohe.shape=}")

        id2label = {i: cls for i, cls in enumerate(mlb.classes_)}
        label2id = {v: k for k, v in id2label.items()}

        # Save run data
        with open(str(loc_p / "id2label.yaml"), "w") as f:
            yaml.dump(id2label, f)

        with open(str(loc_p / "label2id.yaml"), "w") as f:
            yaml.dump(label2id, f)

        X_train, X_test, y_train, y_test = train_test_split(
            ohe, ohe_targets, test_size=0.2
        )

        update("Training...")
        model = OneVsRestClassifier(
            LogisticRegression(n_jobs=6, random_state=0, max_iter=300),
            n_jobs=-1,
        )
        train(model, X_train, X_test, y_train, y_test, loc_p)
        output(mlb, model, X_test, y_test, loc_p / "LogisticRegression.csv")
        dump(model, loc_p / "LogisticRegression.joblib")

        model = OneVsRestClassifier(
            SGDClassifier(n_jobs=8, max_iter=2000, random_state=0),
            n_jobs=-1,
        )
        train(model, X_train, X_test, y_train, y_test, loc_p)
        output(mlb, model, X_test, y_test, loc_p / "SGDClassifier.csv")
        dump(model, loc_p / "SGDClassifier.joblib")

        model = OneVsRestClassifier(
            MultinomialNB(),
            n_jobs=-1,
        )
        train(model, X_train, X_test, y_train, y_test, loc_p)
        output(mlb, model, X_test, y_test, loc_p / "MultinomialNB.csv")
        dump(model, loc_p / "MultinomialNB.joblib")
