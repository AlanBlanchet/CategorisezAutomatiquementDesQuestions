from pathlib import Path
import shutil
from typing import ClassVar, Literal
from bs4 import BeautifulSoup
import nltk
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from pandarallel import pandarallel
import ipywidgets as widgets
from IPython.display import display
import regex as re
import cudf

pd.options.mode.chained_assignment = None
pandarallel.initialize(verbose=0)


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
        self.targets = [f"target{x+1}" for x in range(Dataset.num_targets)]
        self.freq = self.__preprocess()

        self._id2label = {k: v for k, v in enumerate(self.freq.keys())}
        self._label2id = {v: k for k, v in self._id2label.items()}
        self.id2label = lambda keys: list(map(self._id2label.get, keys))
        self.label2id = lambda keys: list(map(self._label2id.get, keys))

    def __preprocess(self):
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

        # Simple parsing
        self.df["text"] = self.df["text"].str.lower().parallel_apply(parse_html)
        self.df["title"] = self.df["title"].str.lower()
        # Keep original for comparison
        self.df["original_text"] = self.df["text"]
        self.df["original_title"] = self.df["title"]
        # Preprocess
        self.df["text"] = (
            self.df["text"]
            .parallel_apply(url_remover)
            .parallel_apply(tokenize)
            .parallel_apply(stemming)
        )
        self.df["title"] = (
            self.df["title"].parallel_apply(url_remover).parallel_apply(tokenize)
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
        # See data examples for the specified version of the script
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

    def __getitem__(self, index):
        return self.df.loc[index]
