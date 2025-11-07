"""
In this module, we want to check that there are no outliers in the training embeddings
that would disturb the model estimation for the healthy distibution.

To do so, we plot UMAP embeddings for the training latent codes.

"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
from matplotlib.patches import Ellipse
from multivae.models import AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from mvae_ad.data.dataset_handler import (
    get_hypo_datasets,
    get_test_datasets,
    get_train_val,
)
from mvae_ad.metrics.metrics import get_hydra_config_and_model
from mvae_ad.metrics.utils import get_embeddings_and_id_dict


def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x, mean_y = mean[0], mean[1]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    print(mean_x, mean_y, scale_x, scale_y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def save_pca_projection_png(
    embeddings, labels, output_path, covs=[], embeddings_for_fit=None, logvars=None
):
    if embeddings_for_fit is None:
        embeddings_for_fit = embeddings

    pca = PCA(n_components=2).fit(embeddings_for_fit)
    pca_embeddings = pca.transform(embeddings)
    fig, ax = plt.subplots(1, 1)
    for name in np.unique(labels):
        if name == 1:
            if logvars is not None and name == 1:
                c = logvars.sum(-1)
            else:
                c = None
            sc = ax.scatter(
                pca_embeddings[labels == name, 0],
                pca_embeddings[labels == name, 1],
                alpha=0.3,
                # label=name,
                c=c,
            )
        else:
            sc = ax.scatter(
                pca_embeddings[labels == name, 0],
                pca_embeddings[labels == name, 1],
                alpha=0.3,
                # label=name,
                color="orange",
            )

    for cov in covs:
        print(cov.location_, cov.covariance_)

        print(pca.components_.T.shape, cov.covariance_.shape, pca.components_.shape)
        pca_cov = pca.components_ @ cov.covariance_ @ pca.components_.T
        # pca_mean = pca.transform(cov.location_)
        confidence_ellipse(pca_cov, [0, 0], ax, n_std=1.0)

    # plt.legend()
    fig.colorbar(sc)
    plt.title(
        f"PCA projection of train and val embeddings (ratio :{pca.explained_variance_ratio_})"
    )
    plt.savefig(Path(output_path) / "pca_projection.png")
    plt.close()


def project_embeddings_2d(model, datasets, batch_size, model_path, methods=["umap"]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_embeddings = []
    labels = []
    for name, dataset in datasets.items():
        # Compute all the embeddings and logvars
        embeddings_dict = get_embeddings_and_id_dict(
            model, dataset, batch_size=batch_size, device=device
        )
        all_embeddings.append(embeddings_dict["embeddings"])
        labels.extend([name] * len(dataset))

    all_embeddings = torch.cat(all_embeddings).cpu().numpy()
    labels = np.array(labels)
    # project in 2d space
    if "umap" in methods:
        trans = UMAP(n_neighbors=15, random_state=42).fit(all_embeddings)

        plt.figure()
        for name in datasets:
            plt.scatter(
                trans.embedding_[labels == name, 0],
                trans.embedding_[labels == name, 1],
                label=name,
            )

        plt.legend()
        plt.title("UMAP projection of train and val embeddings")
        plt.savefig(Path(model_path) / "umap_projection.png")

    if "pca" in methods:
        save_pca_projection_png(all_embeddings, labels, Path(model_path))

    if "tsne" in methods:
        tsne_embeddings = TSNE(n_components=2).fit_transform(all_embeddings)
        plt.figure()
        for name in datasets:
            plt.scatter(
                tsne_embeddings[labels == name, 0],
                tsne_embeddings[labels == name, 1],
                label=name,
            )

        plt.legend()
        plt.title("TSNE projection of train and val embeddings")
        plt.savefig(Path(model_path) / "tsne_projection.png")

    return


if __name__ == "__main__":
    # Parse the model and dataset
    parser = ArgumentParser()
    parser.add_argument("--hydra_path")
    parser.add_argument("--checkpoint_window", default=None)
    args = parser.parse_args()

    params, model_path = get_hydra_config_and_model(
        args.hydra_path, window=args.checkpoint_window
    )

    # Get the model
    best_model = AutoModel.load_from_folder(model_path)

    # Get train, val and hypo datasets
    train_dataset, val_dataset = get_train_val(params)
    val_hypo = get_hypo_datasets(params, params.eval.simulated_datasets)
    test_ad = get_test_datasets(params, params.eval.test_datasets)

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "simu_30": val_hypo,
        "ad": test_ad,
    }

    project_embeddings_2d(
        model=best_model,
        datasets=datasets,
        batch_size=params.batch_size,
        model_path=model_path,
        methods=["umap", "tsne", "pca"],
    )
