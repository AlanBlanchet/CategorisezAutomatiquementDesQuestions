"""
The Dataset module used throughout the project.
This is where generic function to be reused were placed
"""

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import urlparse

import cudf
import datasets
import ipywidgets as widgets
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mlflow
import nltk
import numpy as np
import pandas as pd
import regex as re
import torch
from bs4 import BeautifulSoup
from IPython.display import display
from pandarallel import pandarallel
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    TrainingArguments,
    get_scheduler,
)

from src.models.callback import get_callback
from src.models.metrics import top_k
from src.models.trainer import SOTrainer

pd.options.mode.chained_assignment = None
pandarallel.initialize(verbose=0)
mlflow.autolog(disable=True)
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

mlflow.set_tracking_uri(
    f"file://" + str(Path(__file__).parents[2] / "notebooks/mlruns")
)


@dataclass
class Dataset:
    name: str
    n: int = -1
    device: Literal["gpu", "cpu"] = "cpu"

    DEFAULT_VERSION: ClassVar[int] = 1e20
    data_path: ClassVar[Path] = Path(__file__).parents[2] / "data"
    raw_path: ClassVar[Path] = data_path / "raw"
    processed_path: ClassVar[Path] = data_path / "processed"
    url_regex: ClassVar[
        str
    ] = r"(\w+:\/{2}|www\.)[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*"
    num_targets: ClassVar[int] = 5
    target_encodings: ClassVar[dict] = {
        # c#, f#, j# to c-sharp, f-sharp, j-sharp
        "(c|f|j)#": r"\1-sharp",
        # c++ to cpp
        r"\+\+": "pp",
        # gdi+ to gdi-plus
        r"\+": "-plus",
        # pkcs#12, pkcs#5 to pkcs
        r"(.*)#\d+": r"\1",
    }
    targets: ClassVar[list[str]] = [f"target{x+1}" for x in range(num_targets)]

    def __post_init__(self):
        Dataset.init()

        path = Dataset.raw_path / self.name

        feather_p = path.with_suffix(".arrow")

        if feather_p.exists():
            # Get arrow for fast loading
            df = pd.read_feather(feather_p).iloc[: self.n]
        else:
            # Will be a cudf DataFrame but to get type hints type it as a pandas DataFrame
            df = pd.read_csv(path.with_suffix(".csv"))[["Title", "Body", "Tags"]]
            # Output to arrow for next import
            df.to_feather(feather_p)
            df = df.iloc[: self.n]

        self.df: pd.DataFrame = df
        # Do some basic preprocessing
        self.freq = self.__preprocess()

        self._id2label = {k: v for k, v in enumerate(self.freq.keys())}
        self._label2id = {v: k for k, v in self._id2label.items()}
        self.id2label = lambda keys: list(map(self._id2label.get, keys))
        self.label2id = lambda keys: list(map(self._label2id.get, keys))

    def __preprocess(self):
        """
        Custom preprocessing of the data
        """

        self.df.columns = ["title", "text", "target"]

        def parse_html(text):
            soup = BeautifulSoup(text, "html.parser")
            # code = soup.find_all("code")
            # [c.decompose() for c in code]
            # Return text only
            return soup.get_text(separator=" ")

        def url_remover(text):
            text = re.sub(
                Dataset.url_regex,
                "",
                text,
                flags=re.MULTILINE,
            )
            return text

        def stemming(text):
            text = [Dataset.nltk_stemmer.stem(plural) for plural in text.split(" ")]
            return " ".join(text)

        def tokenize(text):
            text = nltk.word_tokenize(text)
            text = nltk.tokenize.MWETokenizer(
                [("c", "#"), ("f", "#"), ("+", "+")], separator=""
            ).tokenize(text)
            text = [t for t in text if t not in Dataset.nltk_stopwords]
            text = nltk.RegexpTokenizer(r"\w+[#-+]*").tokenize(" ".join(text))
            return " ".join(text)

        # Keep original for comparison
        self.df["original_text"] = self.df["text"].parallel_apply(parse_html)
        self.df["original_title"] = self.df["title"]
        # Simple parsing
        self.df["text"] = self.df["original_text"].str.lower()
        self.df["title"] = self.df["original_title"].str.lower()
        # Preprocess
        self.df["text"] = (
            self.df["text"]
            .parallel_apply(url_remover)
            .parallel_apply(tokenize)
            .parallel_apply(stemming)
        )
        self.df["title"] = (
            self.df["title"]
            .parallel_apply(url_remover)
            .parallel_apply(tokenize)
            .parallel_apply(stemming)
        )
        # Change target encoding
        self.df["target"] = (
            self.df["target"].str.replace("><", "|").str.strip("<>").str.lower()
        )

        # Expand targets
        self.df[self.targets] = (
            self.df["target"]
            .str.split("|", expand=True)
            .loc[:, : Dataset.num_targets - 1]
        )
        self.df = self.df.drop("target", axis="columns")

        def encode(target: str):
            for key, value in Dataset.target_encodings.items():
                if re.search(key, target) is not None:
                    return re.sub(key, value, target)
            return target

        # Encode targets
        self.df[self.targets] = self.df[self.targets].parallel_apply(
            lambda targets: [encode(target) for target in targets.values]
        )

        # Get target frequencies
        f = nltk.FreqDist()
        for x in self.df[self.targets].values.flatten():
            f[x] += 1

        # Sort targets to have most common to target 1 and lowest to target 5
        self.df[self.targets] = self.df[self.targets].parallel_apply(
            lambda df: pd.Series(sorted(df.values, key=lambda x: f[x], reverse=True)),
            axis=1,
        )

        return f

    @classmethod
    def init(cls):
        if not hasattr(cls, "stopwords") or not hasattr(cls, "stemmer") is None:
            csvs = [*cls.data_path.rglob("*.csv")]

            cls.raw_path.mkdir(exist_ok=True)
            cls.processed_path.mkdir(exist_ok=True)

            for i, csv in enumerate(csvs, start=1):
                shutil.move(csv, cls.raw_path / f"topics{i}.csv")

            nltk_path = cls.data_path / ".nltk"

            nltk.download("stopwords", download_dir=nltk_path, quiet=True)
            nltk.download("punkt", download_dir=nltk_path, quiet=True)
            nltk.data.path.append(nltk_path)

            cls.nltk_stopwords = nltk.corpus.stopwords.words("english")
            cls.nltk_stemmer = nltk.stem.PorterStemmer()
        return cls

    def to(self, device: Literal["cpu", "gpu"] = "cpu"):
        """
        Method used to save memory by switching the DataFrame device
        """
        if device == "cpu" and type(self.df) == cudf.DataFrame:
            self.device = device
            self.df = self.df.to_pandas()
        elif device == "gpu" and type(self.df == pd.DataFrame):
            self.device = device
            self.df = cudf.DataFrame(self.df)

    @classmethod
    def use(
        cls,
        name: str,
        source: Literal["raw", "processed"] = "raw",
        n_sample: int = None,
        original=False,
        random_state=None,
        index=None,
        version=DEFAULT_VERSION,
    ):
        cls.init()
        path = cls.raw_path / name
        if source == "processed":
            path = cls.processed_path / name

        if index is not None:
            df = pd.read_csv(path, skiprows=lambda x: x not in [0, index + 1])[
                ["Title", "Body", "Tags"]
            ]
        else:
            df = pd.read_csv(path)[["Title", "Body", "Tags"]]

        if index is None and n_sample is not None:
            df = df.sample(n_sample, random_state=random_state)

        return cls._preprocess(df, original=original, version=version)

    def most_common(self, name, fps=50, skip=100, n=50, animate=False):
        """
        Get the most common tokens
        """
        freq = nltk.FreqDist()

        if animate:
            frames = len(self.df["text"]) // skip
            loader = tqdm(total=frames)
            fig, ax = plt.subplots()
            _, _, patches = ax.hist([], bins=n)
            plt.xticks(rotation=50, ha="right")
            for i, patch in enumerate(patches):
                patch.set_width(1)
                patch.set_x(i)

            def anim(frame):
                for i in range(frame * skip, frame * skip + skip):
                    freq.update(self.df["text"][i].split(" "))

                common = list(zip(*freq.most_common(n)))
                vals = common[0]
                counts = common[1]

                ax.set_ylim(0, max(counts))
                ax.set_xticks(list(range(len(vals))), labels=vals)

                for patch, count in zip(patches, counts):
                    patch.set_height(count)

                ax.set_title(f"Frame {frame}")
                loader.update(1)
                return patches

            hist_anim = animation.FuncAnimation(
                fig,
                anim,
                frames=frames,
                interval=1000 // fps,
                cache_frame_data=False,
                blit=True,
            )
            hist_anim.save(f"{name}.gif")
            loader.close()
            plt.show()
        else:
            [freq.update(sentence.split(" ")) for sentence in tqdm(self.df["text"])]

        return freq

    def example(
        self,
        index=None,
        interactive=True,
        mode: Literal["title", "text"] = "text",
    ):
        """
        See data examples
        """
        if interactive:
            if index is None:
                index = 0
            else:
                index = index
            from_button = False

            def operation(n):
                global index, from_button
                from_button = True

                def op_show(_):
                    show(index + n)

                return op_show

            prev_b = widgets.Button(description="Prev", button_style="danger")
            prev_b.on_click(operation(-1))

            next_b = widgets.Button(description="Next", button_style="success")
            next_b.on_click(operation(1))

            def update_text(x):
                global index
                if x["old"] == index:
                    show(x["new"])

            index_i = widgets.BoundedIntText(
                name="IntInput", value=index, step=1, start=0, end=5e4, max=5e4
            )
            index_i.observe(update_text, "value")

            hbox = widgets.HBox([prev_b, next_b, index_i], background_color="red")

            def show(n):
                global index
                index = n
                index_i.value = index
                display(hbox, clear=True)
                try:
                    print("-" * 15, f"{index=}")
                    self.example(index, mode=mode, interactive=False)
                except:
                    print(f"Error with {index=}")

            show(index)
        else:
            line = self.df.loc[index]

            print("Original " + "=" * 44 + "\n", line[f"original_{mode}"])
            print("Parsed " + "=" * 46 + "\n", str(line[mode]))
            print("Targets " + "=" * 10, line[self.targets].values)

    def to_datasets(
        self,
        targets: list[str] = None,
        mapper: dict = {},
        tokenizer=None,
        sentence_length=50,
    ):
        """
        Convert to hugging face API datasets
        """
        self._tokenizer = tokenizer

        if targets is None:
            targets = self.targets

        counts = self.df[targets[0]].value_counts()
        has_target = self.df[targets[0]].isin(counts[counts > 8].index)

        df = self.df.loc[has_target]
        data = df[targets]

        data["text"] = df["original_title"] + df["original_text"]

        # Transform to the Dataset API
        for target in targets:
            data[target] = self.label2id(data[target].values)

        dataset: datasets.Dataset = datasets.Dataset.from_pandas(data)
        dataset = dataset.rename_columns(mapper)

        if tokenizer:

            def preprocess(batch):
                return tokenizer(
                    batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=sentence_length,
                )

            # PREPROCESS
            dataset = dataset.map(preprocess, batched=True)

        dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels", "token_type_ids"]
        )

        dataset = dataset.class_encode_column("labels")

        dataset = dataset.train_test_split(
            test_size=0.2, seed=1234, stratify_by_column="labels"
        )

        return dataset

    def mlflow_run(self):
        """
        Starts a run and returns a function to call when finished
        """
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.start_run(run_name=str(uuid.uuid4()))
        return mlflow.end_run

    def trainer(
        self,
        model_name,
        dataset: datasets.Dataset,
        epochs=10,
        lr=8e-5,
        batch_size=16,
        gradient_accumulation_steps=16,
        peft: LoraConfig = None,
        save_dataset=True,
    ):
        """
        Get the hugging face trainer
        """
        mlflow_end = self.mlflow_run()

        try:
            loc = Path(urlparse(mlflow.get_tracking_uri()).path) / "../artifacts"
            loc.mkdir(exist_ok=True)
            loc: Path = loc / mlflow.active_run().info.run_id
            loc.mkdir(exist_ok=True)
            if save_dataset:
                for split in ["train", "test"]:
                    df: pd.DataFrame = (
                        dataset[split].to_pandas().rename({"labels": "targets"})
                    )
                    name = (loc / f"{split}.csv").resolve()
                    df.to_csv(name)
                    mlflow_input = mlflow.data.from_pandas(df, source=name)

                    mlflow.log_input(mlflow_input, context=split)
                    mlflow.log_artifact(name)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self._id2label),
                id2label=self._id2label,
                label2id=self._label2id,
            )

            if peft is not None:
                model = get_peft_model(model, peft)
                model.print_trainable_parameters()

            # Create the optimizer
            optimizer = AdamW(model.parameters(), lr=lr)

            # Set the number of training steps
            num_train_steps = (
                len(dataset["train"]) / (batch_size * gradient_accumulation_steps)
            ) * epochs

            # Create the scheduler
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=int(num_train_steps * 0.02),
                # num_warmup_steps=0,
                num_training_steps=num_train_steps,
            )

            torch.set_float32_matmul_precision("medium")

            args = TrainingArguments(
                str(loc),
                num_train_epochs=epochs,
                evaluation_strategy="steps",
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                # eval_delay=2,
                weight_decay=0.03,
                eval_steps=100,
                metric_for_best_model="accuracy",
                save_strategy="epoch"
                # optim="adamw_torch"
            )

            iteration = {"i": 1}

            def compute_metrics(eval_preds):
                logits, labels = eval_preds

                preds = torch.tensor(logits).softmax(dim=-1).numpy()
                top_preds = np.argmax(preds, axis=-1)

                pd.DataFrame(
                    {"target": self.id2label(labels), "pred": self.id2label(top_preds)}
                ).to_csv(loc / "pred_ex{}.csv".format(iteration["i"]))

                iteration["i"] += 1

                metrics = {
                    "accuracy": accuracy_score(labels, top_preds),
                    "precision_weighted": precision_score(
                        labels, top_preds, average="weighted", zero_division=np.nan
                    ),
                    "precision_micro": precision_score(
                        labels, top_preds, average="micro", zero_division=np.nan
                    ),
                    "precision_macro": precision_score(
                        labels, top_preds, average="macro", zero_division=np.nan
                    ),
                    "recall_weighted": recall_score(
                        labels, top_preds, average="weighted"
                    ),
                    "recall_macro": recall_score(labels, top_preds, average="macro"),
                    "f1_score": f1_score(labels, top_preds, average="macro"),
                    **{f"top{i}": top_k(labels, preds, i) for i in range(2, 5)},
                }

                return metrics

            trainer = SOTrainer(
                model,
                args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                compute_metrics=compute_metrics,
                callbacks=[get_callback(mlflow_end)],
                optimizers=[optimizer, scheduler],
            )

            return trainer
        except Exception as e:
            mlflow_end()
            raise e

    def save(self):
        """
        Save id2label as json
        """
        save_p = self.data_path / "id2label.json"
        save_p.write_text(json.dumps(self._id2label))

    def __getitem__(self, index):
        return self.df.loc[index]
