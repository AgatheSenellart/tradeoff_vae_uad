import math
import os
import random

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from mvae_ad.architectures.covariates_networks import (
    CovariatesNetwork,
    JointEncoderNetwork,
    PriorCovariatesNetwork,
)


def seed_everything(seed):
    """reset all the seeds"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_encoder_decoder_and_prior(params: DictConfig, train_dataset):
    """
    Constructs and returns the encoder, decoder, and prior network based on the provided parameters and training dataset.
    Args:
        params (DictConfig): Configuration object containing model and architecture parameters.
        train_dataset: Dataset object containing training data and metadata, including modalities and dimensions.
    Returns:
        Tuple:
            - encoder: The instantiated encoder network, potentially conditioned on covariates.
            - decoder: The instantiated decoder network, potentially conditioned on covariates.
            - prior_network: The instantiated prior network, potentially conditioned on covariates.
    Notes:
        - The function dynamically creates networks based on the configuration flags in `params`.
        - Covariates networks are created for encoder, decoder, and prior networks if the respective flags
          (`cond_in_encoder.use`, `cond_in_decoder.use`, `cond_in_prior.use`) are set to `True`.
        - The input dimensions for covariates networks are derived from the training dataset's modalities and dimensions.
    """

    if hasattr(params.model, "cond_in_encoder") and params.model.cond_in_encoder.use:
        seed_everything(params.seed)

        cond_mod_network = CovariatesNetwork(
            cond_mods=train_dataset.txt_modalities,
            input_dim=2 + train_dataset.dim_dict["age"].numel(),
            hidden_dims=params.model.cond_in_encoder.hidden_dims,
        )

        joint_network = JointEncoderNetwork(
            dim_image=params.architecture.latent_dim,
            dim_covars=cond_mod_network.output_dim,
            hidden_dims=params.model.cond_in_encoder.joint_layers,
        )
    else:
        cond_mod_network = joint_network = None

    seed_everything(params.seed)
    encoder = instantiate(
        params.architecture.encoder,
        latent_dim=params.architecture.latent_dim,
        cond_mod_network=cond_mod_network,
        joint_network=joint_network,
        input_shape=params.data.extraction.dim,  # for MLP architectures
    )

    # Decoder

    if hasattr(params.model, "cond_in_decoder") and params.model.cond_in_decoder.use:
        seed_everything(params.seed)
        cond_mod_decoder = CovariatesNetwork(
            cond_mods=train_dataset.txt_modalities,
            input_dim=2 + train_dataset.dim_dict["age"].numel(),
            hidden_dims=params.model.cond_in_decoder.hidden_dims,
        )
    else:
        cond_mod_decoder = None

    seed_everything(params.seed)
    decoder = instantiate(
        params.architecture.decoder,
        latent_dim=params.architecture.latent_dim,
        cond_mod_network=cond_mod_decoder,
        input_shape=params.data.extraction.dim,  # for MLP architectures
    )

    # Prior Network

    if hasattr(params.model, "cond_in_prior") and params.model.cond_in_prior.use:
        seed_everything(params.seed)
        prior_network = PriorCovariatesNetwork(
            cond_mods=train_dataset.txt_modalities,
            input_dim=2 + train_dataset.dim_dict["age"].numel(),
            latent_dim=params.architecture.latent_dim,
            hidden_dims=params.model.cond_in_prior.hidden_dims,
        )
    else:
        prior_network = None

    # Metric network for the RHVAE
    if hasattr(params.model, "metric_network"):
        if params.model.config.uses_covariance_as_metric:
            metric_network = None
        else:
            metric_network = instantiate(
                params.model.metric_network,
                input_dim=math.prod(params.data.extraction.dim),
                latent_dim=params.architecture.latent_dim,
            )
    else:
        metric_network = None

    return encoder, decoder, prior_network, metric_network


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
