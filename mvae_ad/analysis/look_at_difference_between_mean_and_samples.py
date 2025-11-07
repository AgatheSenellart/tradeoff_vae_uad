"""
In this file, we investigate the difference when fitting p_healthy on the mean embeddings or
on posterior samples.

We compute the covariance in each case and compare them.

"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from multivae.models import AutoModel
from sklearn.covariance import EmpiricalCovariance

from mvae_ad.data.dataset_handler import get_train_val
from mvae_ad.metrics.compare_embedding_dists import wasserstein_between_gaussians
from mvae_ad.metrics.metrics import get_hydra_config_and_model
from mvae_ad.metrics.utils import get_embeddings_and_id_dict

# create logger
logger = logging.getLogger(__name__)

# create console handler and set level to debug
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def compare_fit_on_mean_and_fit_on_samples(
    training_means,
    training_samples,
    val_means,
    val_samples,
    output_dir,
):
    dict_emb = {"means": training_means, "samples": training_samples}
    dict_covs = {}
    dict_mus = {}

    for name, embeds in dict_emb.items():
        # Estimate mean and covariance of training_embeddings for the prior distribution of flows
        cov = EmpiricalCovariance()
        cov.fit(embeds)

        # Initial Gaussian likelihood
        ll_gaussian_val_mean = cov.score(val_means.detach().cpu().numpy()).sum()
        logger.info(
            "Fit on %s, Gaussian likelihood on val means: %s",
            name,
            ll_gaussian_val_mean,
        )
        ll_gaussian_val_samples = cov.score(val_samples.detach().cpu().numpy()).sum()
        logger.info(
            "Fit on %s, Gaussian likelihood on val samples: %s",
            name,
            ll_gaussian_val_samples,
        )

        ll_gaussian_train_mean = cov.score(training_means.detach().cpu().numpy()).sum()
        logger.info(
            "Fit on %s, Gaussian likelihood on train means: %s",
            name,
            ll_gaussian_train_mean,
        )
        ll_gaussian_train_samples = cov.score(
            training_samples.detach().cpu().numpy()
        ).sum()
        logger.info(
            "Fit on %s, Gaussian likelihood on train samples: %s",
            name,
            ll_gaussian_train_samples,
        )

        dict_covs[name] = cov.covariance_
        dict_mus[name] = cov.location_

    # Compute KL and Wasserstein distance between the two distributions.
    wss = wasserstein_between_gaussians(
        dict_mus["samples"], dict_mus["means"], dict_covs["samples"], dict_covs["means"]
    )

    logger.info("Wasserstein between distributions : %s", wss)

    # Plot the means and covariances
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    im = ax[0].imshow(dict_mus["samples"].reshape(16, 16))
    fig.colorbar(im, ax=ax[0])
    im1 = ax[1].imshow(dict_mus["means"].reshape(16, 16))
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow((dict_mus["samples"] - dict_mus["means"]).reshape(16, 16))
    fig.colorbar(im2, ax=ax[2])
    fig.savefig(Path(output_dir) / "means.png")

    #
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    im = ax[0].imshow(dict_covs["samples"])
    fig.colorbar(im, ax=ax[0])
    im1 = ax[1].imshow(dict_covs["means"])
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow((dict_covs["samples"] - dict_covs["means"]))
    fig.colorbar(im2, ax=ax[2])
    fig.savefig(Path(output_dir) / "covs.png")


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

    # Compute all the training embeddings and logvars
    train_embeddings_dict = get_embeddings_and_id_dict(
        best_model, train_dataset, batch_size=params.batch_size, device="cpu"
    )
    train_embeddings, train_logvars = (
        train_embeddings_dict["embeddings"],
        train_embeddings_dict["logvars"],
    )
    train_samples = torch.distributions.Normal(
        train_embeddings, torch.exp(0.5 * train_logvars)
    ).sample()

    val_embeddings_dict = get_embeddings_and_id_dict(
        best_model, val_dataset, params.batch_size, "cpu"
    )["embeddings"]

    val_embeddings, val_logvars = (
        val_embeddings_dict["embeddings"],
        val_embeddings_dict["logvars"],
    )
    val_samples = torch.distributions.Normal(
        val_embeddings, torch.exp(0.5 * val_logvars)
    ).sample()

    output_path = Path(model_path) / "compare_fit_on_means or samples"

    # For local debugging
    # train_embeddings = torch.distributions.Normal(0,1).sample([400,256])
    # train_samples = torch.distributions.Normal(0,1).sample([400,256])

    # val_embeddings = torch.distributions.Normal(0,1).sample([200,256])
    # val_samples = torch.distributions.Normal(0,1).sample([200,256])
    # output_path = 'test'

    Path(output_path).mkdir(exist_ok=True)

    compare_fit_on_mean_and_fit_on_samples(
        train_embeddings, train_samples, val_embeddings, val_samples, output_path
    )
