"""This script is to look at the distributions of the variances given by the encoder
on the validation set.
The goal is to show that we obtain larger variations with the sparse VAE which results in
a smoother latent space.
"""

import sys
import traceback
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from multivae.models import AutoModel
from omegaconf import DictConfig

from mvae_ad.data.dataset_handler import get_brain_masks, get_train_val
from mvae_ad.metrics.metrics import get_hydra_config_and_model
from mvae_ad.metrics.utils import get_embeddings_and_id_dict
from mvae_ad.model.utils import seed_everything


def compute_posteriors_variances(eval_params: DictConfig):
    print("We're comparing reconstruction methods!")
    # set the seed
    seed_everything(eval_params.seed)

    training_params, model_path = get_hydra_config_and_model(
        eval_params.hydra_path, eval_params.checkpoint_window
    )
    model_path = Path(model_path)

    # set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get the model
    best_model = AutoModel.load_from_folder(model_path)
    best_model = best_model.eval().to(device)

    # Get the datasets
    _, val_set = get_train_val(training_params)

    # Get the masks
    brain_mask, hypo_mask = get_brain_masks(training_params)
    brain_mask = brain_mask.to(device)
    hypo_mask = hypo_mask.to(device)

    # Compute all the val embeddings and logvars
    val_embeddings_dict = get_embeddings_and_id_dict(
        best_model, val_set, batch_size=training_params.batch_size, device=device
    )

    # For each dimension, store the posterior variances
    dict_variances = {"sigmas": [], "nsigmas": [], "mus": []}
    for i in range(val_embeddings_dict["embeddings"].shape[-1]):
        sigmas = val_embeddings_dict["logvars"][:, i].mul(0.5).exp()
        mus = val_embeddings_dict["embeddings"][:, i].detach().cpu().numpy()
        dict_variances["sigmas"].extend(list(sigmas.detach().cpu().numpy()))
        dict_variances["nsigmas"].extend(
            list(sigmas.detach().cpu().numpy() / np.std(mus))
        )
        dict_variances["mus"].extend(list(mus))

    df = pd.DataFrame(dict_variances)
    df.to_csv(model_path / "posterior_variances.csv", float_format="{:.3e}".format)


@hydra.main(
    version_base=None,
    config_path="../configs/configs_eval_scripts",
    config_name="config",
)
def main(eval_params: DictConfig) -> None:
    """
    Main function with a more suited handling of errors in order
    for them to be logged in the .out log files even when launched with submitit."""

    try:
        compute_posteriors_variances(eval_params=eval_params)
    except BaseException:
        print("We're in BaseException")
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
