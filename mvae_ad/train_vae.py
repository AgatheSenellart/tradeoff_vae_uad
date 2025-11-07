import logging
import math
import sys
import traceback
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from multivae.models import CVAE
from omegaconf import DictConfig

from mvae_ad.data.dataset_handler import (
    get_brain_masks,
    get_hypo_datasets,
    get_test_datasets,
    get_train_val,
)
from mvae_ad.metrics.compare_embedding_dists import compute_all_wasserstein_dists
from mvae_ad.metrics.metrics import (
    get_eval_statistics,
    get_test_dataset_metrics,
    get_train_z_statistics,
)
from mvae_ad.model.utils import (
    count_parameters,
    get_encoder_decoder_and_prior,
    seed_everything,
)
from mvae_ad.trainers import CapsTrainer
from mvae_ad.trainers.utils import setup_logger


def train_vae(params: DictConfig) -> None:
    """
    Contains all the training in this function.
    Hydra acts like a parser : we get all the arguments from the config file
    (or overriden in command line) in a dictionary.
    """

    PROJECT_NAME = "VAE_UAD"

    logger = logging.Logger("__name__")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # For seed_equal_to_split, change the random seed:
    if params.seed_equal_to_split:
        params.seed = params.data.split + 30
        logger.info(f"Reset seed to split+30 = {params.seed}")

    # Get the datasets
    (
        train_dataset,
        val_dataset,
    ) = get_train_val(params)

    hypo_datasets = get_hypo_datasets(params, params.eval.simulated_datasets)
    test_datasets = get_test_datasets(params, params.eval.test_datasets)

    # Get the masks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brain_mask, hypo_mask = get_brain_masks(params)
    brain_mask = brain_mask.to(device)
    hypo_mask = hypo_mask.to(device)

    # Set the model configuration
    model_config = instantiate(
        params.model.config,
        input_dims={
            "pet_linear": tuple(params.data.extraction.dim),
            **train_dataset.dim_dict,
        },
        latent_dim=params.architecture.latent_dim,
        input_dim=math.prod(params.data.extraction.dim),
    )

    # Get the encoder, decoder and prior_network
    encoder, decoder, prior_network, metric_network = get_encoder_decoder_and_prior(
        params, train_dataset
    )

    print(model_config)

    if params.model.name == "cvae":
        model_class = CVAE
    else:
        raise NotImplementedError(f"Model {params.model.name} not implemented")

    seed_everything(params.seed)
    model = model_class(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder,
        prior_network=prior_network,
        metric_network=metric_network,
    )

    n_params = count_parameters(model)
    print("Number of trainable parameters : ", n_params)

    if not isinstance(params.batch_size, int):
        batch_size = instantiate(params.batch_size)
    else:
        batch_size = params.batch_size

    trainer_config = instantiate(
        params.trainer.config,
        output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=params.seed,  # override
    )

    logger = setup_logger(params, trainer_config, model_config, PROJECT_NAME)

    trainer = CapsTrainer(
        model=model,
        training_config=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[logger],
    )

    trainer.train()

    ################################################################################
    # Evaluation
    ################################################################################

    model_path = Path(trainer.training_dir, "final_model")
    best_model = trainer._best_model.eval()

    # evaluation on val dataset
    val_stats = get_eval_statistics(
        dataset=val_dataset,
        model=best_model,
        brain_mask=brain_mask,
        hypo_mask=hypo_mask,
        save_dir=model_path / "results_on_val",
        reconstruction_method=None,
        device=device,
        batch_size=params.batch_size,
        save_recon=0,
        save_z=0,
        save_stats=params.eval.save_stats,
    )

    # evaluation on hypo
    z_stats = {}
    for name, dataset in hypo_datasets.items():
        z_stats[name] = get_test_dataset_metrics(
            dataset=dataset,
            model=best_model,
            brain_mask=brain_mask,
            hypo_mask=hypo_mask,
            val_dataset_stats=list(val_stats[:3]),
            save_dir=model_path / name,
            dataset_is_simulated=True,
            save_recon=0,
            save_snippets=3,
            save_z=0,
            device=device,
            batch_size=params.batch_size,
            save_stats=params.eval.save_stats,
        )
    # evaluation on test sets
    for name, dataset in test_datasets.items():
        z_stats[name] = get_test_dataset_metrics(
            dataset=dataset,
            model=best_model,
            brain_mask=brain_mask,
            hypo_mask=hypo_mask,
            val_dataset_stats=list(val_stats[:3]),
            save_dir=model_path / name,
            dataset_is_simulated=False,
            save_recon=0,
            save_snippets=0,
            save_z=0,
            device=device,
            batch_size=params.batch_size,
            save_stats=params.eval.save_stats,
        )

    z_stats["val"] = val_stats[-2:]
    z_stats["train"] = get_train_z_statistics(
        train_dataset,
        best_model,
        brain_mask,
        save_dir=model_path / "train_stats",
        device=device,
        batch_size=params.batch_size,
        save_stats=params.eval.save_stats,
    )

    # Compute wasserstein distances between latent distributions
    compute_all_wasserstein_dists(z_stats, model_path)


@hydra.main(
    version_base=None, config_path="configs/configs_training", config_name="config"
)
def main(params: DictConfig) -> None:
    """
    Main function with a more suited handling of errors in order
    for them to be logged in the .out log files even when launched with submitit."""

    try:
        train_vae(params)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
