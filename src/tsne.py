from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
from tsne_torch import TorchTSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def tsne(X, y, labels, show=True):
    print("Calculating TSNE")
    proj = TorchTSNE(n_iter_callback=1, perplexity=30, n_iter=800)
    pred = proj.fit_transform(torch.tensor(X, device="cuda:0").float())

    if show:
        print("Printing")
        sns.scatterplot(x=pred[:, 0], y=pred[:, 1], color=y, hue=labels)

    print("Calculating KMEANS")
    cls = KMeans(n_clusters=len(set(labels)), n_init=100, random_state=0)
    cls.fit(pred)

    score = adjusted_rand_score(labels, cls.labels_)

    return pred, score


def tsne_anim(name, X, y, labels):
    animation_p = Path("../animations/").resolve()
    print("Out dir set to ", animation_p)
    animation_p.mkdir(exist_ok=True)

    output_p = animation_p / name

    output_p.mkdir(exist_ok=True)

    XX = X
    yy = y

    n = len(yy)

    fig, ax = plt.subplots()
    s = sns.scatterplot(x=np.zeros(n), y=np.zeros(n), color=yy, hue=labels, ax=ax)
    scatter = s.findobj(PathCollection)[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()

    bar = tqdm(range(30, 31, 5))

    for perplexity in bar:
        n_iter = 400
        gif_p = output_p / f"iter{n_iter}-perp{perplexity}.gif"

        if gif_p.exists():
            continue

        it_data = []

        def tensor_callback(i, preds: torch.Tensor):
            bar.set_description(f"TSNE iteration {i}")
            preds = MinMaxScaler(copy=False).fit_transform(preds.cpu().numpy())
            it_data.append(preds)

        bar.set_description("Running TSNE")
        proj = TorchTSNE(
            callbacks=[tensor_callback],
            n_iter_callback=1,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=0,
        )
        proj.fit_transform(torch.tensor(X.todense()).float())

        def callback(i, preds):
            fig.suptitle(f"TSNE iter {i}")
            scatter.set_offsets(preds[i])
            return (scatter,)

        ani = FuncAnimation(
            fig,
            callback,
            frames=n_iter,
            interval=1,
            blit=False,
            repeat=False,
            fargs=(it_data,),
        )

        bar.set_description(f"Outputing to {gif_p}")
        ani.save(gif_p, fps=10)

        pred = it_data[-1]

        cls = KMeans(n_clusters=len(set(labels)), n_init=100, random_state=0)
        cls.fit(pred)

        score = adjusted_rand_score(labels, cls.labels_)

        print(f"ARI {score=}")
