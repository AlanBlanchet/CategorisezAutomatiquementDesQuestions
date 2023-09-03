"""
This module was used for testing different simple sklearn approches.
It utilizes the GridSearchCV to find the best params
"""

import pandas as pd
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, jaccard_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from src.dataset.dataset import Dataset


def train(model, params, X_train, X_test, y_train, y_test):
    name = type(model).__name__

    cv = GridSearchCV(
        model,
        n_jobs=8,
        param_grid=params,
        scoring={
            "accuracy": "accuracy",
            "jaccard": make_scorer(jaccard_score, average="weighted"),
        },
        refit="accuracy",
        cv=None,
    )
    cv.fit(X_train, y_train)
    model = cv.best_estimator_

    pd.DataFrame(cv.cv_results_)[
        ["params", "mean_test_accuracy", "rank_test_accuracy"]
    ].sort_values(by="rank_test_accuracy").to_csv(f"src/run/{name}.csv")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred, average="weighted", zero_division=1)
    print(f"{name:<24} {accuracy=:<10.3f} {jaccard=:<10.3f}")

    return model


if __name__ == "__main__":
    with tqdm(total=5, leave=False, position=0) as pbar:
        update = lambda x, y=1: (pbar.update(y), pbar.set_description(x))

        update("Import dataset", 0)
        topics = Dataset("topics1", n=-1)

        update("Preprocessing / building vocab")
        counts = topics.df["target1"].value_counts()
        has_target = topics.df["target1"].isin(counts[counts > 8].index)

        df = topics.df[has_target]

        text = df["title"] + df["text"]
        targets = topics.label2id(df["target1"])

        update("Vectorizing")
        vectorizer = TfidfVectorizer(min_df=0.02, max_df=0.99)
        ohe = vectorizer.fit_transform(text)

        print(f"Dataset targets={len(targets)} vectors={ohe.shape=}")

        X_train, X_test, y_train, y_test = train_test_split(
            ohe, targets, stratify=targets, random_state=0
        )

        update("Training...")
        model = LogisticRegression(n_jobs=10, random_state=0, max_iter=500)
        train(
            model,
            {
                "max_iter": [100, 300],
                "solver": ["lbfgs", "newton-cholesky"],
            },
            X_train,
            X_test,
            y_train,
            y_test,
        )

        model = SGDClassifier(n_jobs=8, max_iter=2000, random_state=0)
        train(
            model,
            {
                "alpha": [1e-4, 1e-5, 1e-6],
                "tol": [1e-3, 1e-2],
                "penalty": ["l1", "l2", "elasticnet"],
            },
            X_train,
            X_test,
            y_train,
            y_test,
        )

        model = MultinomialNB()
        train(
            model,
            {},
            X_train,
            X_test,
            y_train,
            y_test,
        )

        model = MLPClassifier(max_iter=750, random_state=0)
        train(
            model,
            {
                "hidden_layer_sizes": [100, 500],
                "solver": ["lbfgs", "adam"],
            },
            X_train,
            X_test,
            y_train,
            y_test,
        )
