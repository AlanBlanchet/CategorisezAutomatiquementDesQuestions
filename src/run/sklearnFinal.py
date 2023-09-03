"""
The module used to output the best model after training.
To be used in the front app.
"""

from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset.dataset import Dataset
from src.models.metrics import top_k

export_to = (Dataset.data_path / "../src/front/small").resolve()
export_to.mkdir(exist_ok=True)


if __name__ == "__main__":
    with tqdm(total=5, leave=False, position=0) as pbar:
        update = lambda x, y=1: (pbar.update(y), pbar.set_description(x))

        update("Import dataset", 0)
        topics = Dataset("topics1", n=-1)

        update("Preprocessing / building vocab")
        counts = topics.df["target1"].value_counts()
        has_target = topics.df["target1"].isin(counts[counts > 8].index)

        df = topics.df[has_target].reset_index()

        text = df["title"] + df["text"]
        label2id = {l: i for i, l in enumerate(df["target1"].unique())}
        targets = [label2id[t] for t in df["target1"]]

        update("Vectorizing")
        vectorizer = TfidfVectorizer(min_df=1e-5, max_df=0.99)
        ohe = vectorizer.fit_transform(text)

        print(
            f"Dataset targets={len(targets)} unique={len(set(targets))} vectors={ohe.shape=}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            ohe, targets, stratify=targets, random_state=0
        )

        update("Training")
        model = SGDClassifier(
            n_jobs=8, max_iter=2000, random_state=0, alpha=1e-4, penalty="l2", tol=1e-3
        )
        model.fit(X_train, y_train)

        # Used to get probas
        calibrator = CalibratedClassifierCV(model, cv="prefit", n_jobs=8)
        model = calibrator.fit(X_train, y_train)

        update("Predicting")
        y_pred = model.predict(X_test)
        y_pred_probas = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred, average="weighted", zero_division=1)
        top4 = top_k(y_test, y_pred_probas, 4)

        print(f"{'SGDClassifier':<24} {accuracy=:<10.3f} {jaccard=:<10.3f} {top4=:.3f}")

        export = lambda x: str(export_to / x)

        dump(model, export("model.joblib"))
        dump(vectorizer, export("vectorizer.joblib"))
        dump({v: k for k, v in label2id.items()}, export("id2label.joblib"))
