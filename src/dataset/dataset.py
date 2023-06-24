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

pd.options.mode.chained_assignment = None
pandarallel.initialize(verbose=0)


@dataclass
class Dataset:
    DEFAULT_VERSION: ClassVar[int] = 1e20
    data_path: ClassVar[Path] = Path(__file__).parents[2] / "data"
    raw_path: ClassVar[Path] = data_path / "raw"
    processed_path: ClassVar[Path] = data_path / "processed"

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

            cls.stopwords = nltk.corpus.stopwords.words("english")
            cls.stemmer = nltk.stem.PorterStemmer()
        return cls

    @classmethod
    def _preprocess(cls, df, original=False, version=DEFAULT_VERSION):
        # Rename cols
        df.columns = ["title", "text", "target"]

        def parse_html(text):
            soup = BeautifulSoup(text, "html.parser")
            # Code blocks usually cause trouble
            [code.decompose() for code in soup.find_all("code")]
            # Return text only
            return soup.get_text(separator=" ")

        # Keep original for comparison - Remove html
        if original:
            df["original"] = df["text"].parallel_apply(parse_html)

        if version > 0:

            def stemming(text):
                text = [cls.stemmer.stem(plural) for plural in text.split(" ")]
                return " ".join(text)

            def tokenize(text):
                text = nltk.word_tokenize(text)
                text = [t for t in text if t not in cls.stopwords]
                if version > 1:
                    text = nltk.RegexpTokenizer(r"\w+").tokenize(" ".join(text))
                return " ".join(text)

            # Features
            df["text"] = (
                df["text"]
                .str.lower()
                .parallel_apply(parse_html)
                .parallel_apply(tokenize)
                .parallel_apply(stemming)
            )

        # Target
        df["target"] = df["target"].str.replace("><", "|").str.strip("<>").str.lower()

        return df

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

    @classmethod
    def most_common(
        cls, name: str, version=DEFAULT_VERSION, fps=50, skip=100, n=50, animate=False
    ):
        data = cls.use("topics1.csv", version=version)
        freq = nltk.FreqDist()

        if animate:
            frames = len(data.text) // skip
            loader = tqdm(total=frames)
            fig, ax = plt.subplots()
            _, _, patches = ax.hist([], bins=n)
            plt.xticks(rotation=50, ha="right")
            for i, patch in enumerate(patches):
                patch.set_width(1)
                patch.set_x(i)

            def anim(frame):
                for i in range(frame * skip, frame * skip + skip):
                    freq.update(data.text[i].split(" "))

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
            [freq.update(sentence.split(" ")) for sentence in tqdm(data.text)]

        return freq

    @classmethod
    def example(
        cls,
        name: str,
        version=DEFAULT_VERSION,
        random_state=None,
        index=None,
        interactive=False,
    ):
        # See data examples for the specified version of the script
        if interactive:
            index = 0

            def operation(n):
                global index

                def op_show(x):
                    show(index + n)

                return op_show

            prev_b = widgets.Button(description="Prev", button_style="danger")
            prev_b.on_click(operation(-1))

            next_b = widgets.Button(description="Next", button_style="success")
            next_b.on_click(operation(1))

            def update_text(x):
                if x["old"] != x["new"]:
                    show(x["new"])

            index_i = widgets.BoundedIntText(
                name="IntInput", value=index, step=1, start=0, end=5e4
            )
            index_i.observe(update_text, "value")

            hbox = widgets.HBox([prev_b, next_b, index_i])

            def show(n):
                global index
                index = n
                index_i.value = index
                display(hbox, clear=True)
                try:
                    print("-" * 15, f"{index=}")
                    print(
                        Dataset.example(
                            "topics1.csv", random_state=0, version=version, index=index
                        )
                    )
                except:
                    print(f"Error with {index=}")

            show(0)
        else:
            dataset = cls.use(
                name,
                n_sample=1,
                original=True,
                random_state=random_state,
                version=version,
                index=index,
            )

            print("Original " + "=" * 44 + "\n", dataset["original"].values[0])

            print("Parsed " + "=" * 46 + "\n", str(dataset["text"].values[0]))
            print("Targets " + "=" * 10, dataset["target"].values[0])
