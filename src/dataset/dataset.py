from pathlib import Path
import shutil
from typing import ClassVar, Literal
from bs4 import BeautifulSoup
import nltk
from dataclasses import dataclass
import pandas as pd

pd.options.mode.chained_assignment = None


@dataclass
class Dataset:
    data_path: ClassVar[Path] = Path(__file__).parents[2] / "data"
    raw_path: ClassVar[Path] = data_path / "raw"
    processed_path: ClassVar[Path] = data_path / "processed"

    @classmethod
    def organize(cls):
        csvs = [*cls.data_path.rglob("*.csv")]

        cls.raw_path.mkdir(exist_ok=True)
        cls.processed_path.mkdir(exist_ok=True)

        for i, csv in enumerate(csvs, start=1):
            shutil.move(csv, cls.raw_path / f"topics{i}.csv")

        nltk_path = cls.data_path / ".nltk"

        nltk.download("stopwords", download_dir=nltk_path, quiet=True)
        nltk.download("punkt", download_dir=nltk_path, quiet=True)
        nltk.data.path.append(nltk_path)

        cls.stopwords = nltk.corpus.stopwords.words("english")
        cls.stemmer = nltk.stem.PorterStemmer()
        return cls

    @classmethod
    def _preprocess(cls, df, original=False):
        # Rename cols
        df.columns = ["title", "text", "target"]

        def parse_html(text):
            soup = BeautifulSoup(text, "html.parser")
            # Code blocks usually cause trouble
            [code.decompose() for code in soup.find_all("code")]
            # Return text only
            return soup.get_text(separator=" ")

        def stemming(text):
            text = [cls.stemmer.stem(plural) for plural in text.split(" ")]
            return " ".join(text)

        def tokenize(text):
            text = nltk.word_tokenize(text)
            text = [t for t in text if t not in cls.stopwords]
            return " ".join(text)

        # Keep original for comparison - Removev html
        if original:
            df["original"] = df["text"].apply(parse_html)

        # Features
        df["text"] = (
            df["text"].str.lower().apply(parse_html).apply(tokenize).apply(stemming)
        )

        # Target
        df["target"] = df["target"].str.replace("><", ";").str.strip("<>").str.lower()

        return df

    @classmethod
    def use(
        cls,
        name: str,
        source: Literal["raw", "processed"] = "raw",
        n_sample: int = None,
        original=False,
    ):
        path = cls.raw_path / name
        if source == "processed":
            path = cls.processed_path / name

        df = pd.read_csv(path)[["Title", "Body", "Tags"]]

        if n_sample is not None:
            df = df.sample(n_sample)

        return cls._preprocess(df, original=original)
