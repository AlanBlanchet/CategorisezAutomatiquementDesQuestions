import re
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import nltk
from dash.dependencies import Input, Output
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from src.front.ui import get_app

folder = Path(__file__).parent

nltk_path = folder / ".nltk"

nltk.download("stopwords", download_dir=nltk_path, quiet=True)
nltk.download("punkt", download_dir=nltk_path, quiet=True)
nltk.data.path.append(nltk_path)

nltk_stopwords = nltk.corpus.stopwords.words("english")
nltk_stemmer = nltk.stem.PorterStemmer()


def get_tokens(text: str):
    def url_remover(text):
        text = re.sub(
            r"(\w+:\/{2}|www\.)[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*",
            "",
            text,
            flags=re.MULTILINE,
        )
        return text

    def stemming(text):
        text = [nltk_stemmer.stem(plural) for plural in text.split(" ")]
        return " ".join(text)

    def tokenize(text):
        text = nltk.word_tokenize(text)
        text = nltk.tokenize.MWETokenizer(
            [("c", "#"), ("f", "#"), ("+", "+")], separator=""
        ).tokenize(text)
        text = [t for t in text if t not in nltk_stopwords]
        text = nltk.RegexpTokenizer(r"\w+[#-+]*").tokenize(" ".join(text))
        return " ".join(text)

    text = text.lower()
    text = url_remover(text)
    text = tokenize(text)
    text = stemming(text)
    return text


app = get_app()

importer = lambda x: str(folder / f"small/{x}")

model: SGDClassifier = load(importer("model.joblib"))
vectorizer: TfidfVectorizer = load(importer("vectorizer.joblib"))
id2label: dict[int, str] = load(importer("id2label.joblib"))


@app.callback(
    Output("container", "children"),
    [Input("predict", "n_clicks")],
    [Input("title", "value"), Input("text", "value")],
    prevent_initial_call=True,
)
def update_title(n_clicks, title, text):
    ctx = dash.ctx
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "predict.n_clicks":
        tokens = get_tokens(text)

        print(tokens)

        ohe = vectorizer.transform([tokens])

        print(f"{ohe.shape=}")

        preds = model.predict_proba(ohe)[0]

        preds = preds.argsort()[::-1][:5]

        return [
            dbc.Badge(children=id2label[tag], style={"margin": "10px"}) for tag in preds
        ]


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
