"""This file is for loading a trained model and comparing the latent distributions of healthy embeddings
and abnormal embeddings."""

import logging
import sys
import traceback
from pathlib import Path

import hydra
import pandas as pd
import torch
from multivae.models import AutoModel
from omegaconf import DictConfig

from mvae_ad.data.dataset_handler import (
    get_hypo_datasets,
    get_test_datasets,
    get_train_val,
)
from mvae_ad.metrics.compare_embedding_dists import (
    RBF,
    MMDLoss,
    wasserstein_between_gaussians_samples,
)
from mvae_ad.metrics.metrics import get_hydra_config_and_model
from mvae_ad.metrics.utils import get_embeddings_and_id_dict

# create logger
logger = logging.getLogger(__name__)

# create console handler and set level to debug
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def compute_metrics(params: DictConfig):
    # load the hydra config
    params, model_path = get_hydra_config_and_model(
        params.hydra_path, window=params.checkpoint_window
    )

    # set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model
    best_model = AutoModel.load_from_folder(model_path)
    best_model = best_model.eval().to(device)

    # Get train and val datasets
    train_dataset, val_dataset = get_train_val(params)
    val_hypo_dataset = get_hypo_datasets(params, params.eval.simulated_datasets)
    test_ad_dataset = get_test_datasets(params, params.eval.test_datasets)

    # Compute all the training embeddings and logvars
    train_embeddings_dict = get_embeddings_and_id_dict(
        best_model, train_dataset, batch_size=params.batch_size, device=device
    )
    train_embeddings = train_embeddings_dict["embeddings"]

    # Get the val embeddings
    val_embeddings_dict = get_embeddings_and_id_dict(
        best_model, val_dataset, batch_size=params.batch_size, device=device
    )
    val_embeddings = val_embeddings_dict["embeddings"]

    # Get the hypo_simulated embeddings
    hypo_embeddings_dict = get_embeddings_and_id_dict(
        best_model, val_hypo_dataset, batch_size=params.batch_size, device=device
    )
    hypo_embeddings = hypo_embeddings_dict["embeddings"]

    # Get the test_AD embeddings
    ad_embeddings = get_embeddings_and_id_dict(
        best_model, test_ad_dataset, batch_size=params.batch_size, device=device
    )["embeddings"]

    # Dummy embeddings for local debugging
    # train_embeddings = torch.distributions.Normal(0,1).sample([600,64])
    # val_embeddings = torch.distributions.Normal(0,1).sample([60, 64])
    # hypo_embeddings = torch.distributions.Normal(1,1.2).sample([60, 64])
    # ad_embeddings = torch.distributions.Normal(1.1,1.2).sample([60, 64])

    distance_results = {
        "datasets": ["train_val", "val_hypo", "val_test_ad"],
    }
    # Compute the wasserstein distances
    for cov_estimator in ["EmpiricalCovariance"]:
        w_train_val = wasserstein_between_gaussians_samples(
            train_embeddings.cpu().numpy(),
            val_embeddings.cpu().numpy(),
            covariance_estimator=cov_estimator,
        )
        w_val_hypo = wasserstein_between_gaussians_samples(
            val_embeddings.cpu().numpy(),
            hypo_embeddings.cpu().numpy(),
            covariance_estimator=cov_estimator,
        )
        w_test_ad_val = wasserstein_between_gaussians_samples(
            val_embeddings.cpu().numpy(),
            ad_embeddings.cpu().numpy(),
            covariance_estimator=cov_estimator,
        )
        distance_results[f"wasserstein_{cov_estimator}"] = [
            w_train_val,
            w_val_hypo,
            w_test_ad_val,
        ]

    # Compute MMD distances between the distributions

    # Set the base bandwidth by taking the median or the mean of distances

    for base_bandwidth_selection in ["median"]:
        joint_data = torch.vstack(
            [train_embeddings, val_embeddings, hypo_embeddings, ad_embeddings]
        )
        L2_distances = torch.cdist(joint_data, joint_data) ** 2

        if base_bandwidth_selection == "mean":
            n_samples = L2_distances.shape[0]
            bandwidth = L2_distances.sum() / (
                n_samples * (n_samples - 1)
            )  # remove diagonal terms from the mean
        elif base_bandwidth_selection == "median":
            bandwidth = L2_distances.median()
        else:
            raise AttributeError()

        logger.info("%s bandwidth : %s", base_bandwidth_selection, bandwidth)

        for mul_factor in [2.0]:
            for n_kernel in [5]:
                kernel = RBF(n_kernel, mul_factor, bandwidth=bandwidth)

                mmd = MMDLoss(kernel=kernel)

                mmd_train_val = mmd(train_embeddings, val_embeddings).item()
                mmd_val_hypo = mmd(val_embeddings, hypo_embeddings).item()
                mmd_val_ad = mmd(val_embeddings, ad_embeddings).item()
                distance_results[
                    f"mmd_{base_bandwidth_selection}_bw_{mul_factor}_factor_{n_kernel}_kernels"
                ] = [mmd_train_val, mmd_val_hypo, mmd_val_ad]

    # Saves metrics in a csv
    df = pd.DataFrame(distance_results)
    df.to_csv(Path(model_path) / "distance_distributions.csv")


@hydra.main(
    version_base=None, config_path="configs/configs_eval_scripts", config_name="config"
)
def main(params: DictConfig) -> None:
    """
    Main function with a more suited handling of errors in order
    for them to be logged in the .out log files even when launched with submitit."""

    try:
        compute_metrics(params)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
