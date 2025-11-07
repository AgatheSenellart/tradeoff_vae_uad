"""
In this file, we reload a sparse vae model and we look at the parameter $alpha$
to see how many dimensions are really active.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from multivae.models import AutoModel

from mvae_ad.metrics.metrics import get_hydra_config_and_model

# create logger
logger = logging.getLogger(__name__)

# create console handler and set level to debug
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def get_p_dropouts(log_alpha):
    log_alpha = log_alpha.flatten()
    alpha = torch.exp(log_alpha)
    p = alpha / (1 + alpha)
    return p.detach().numpy()


def sparse_active_dimensions(model, output_dir):
    p = get_p_dropouts(model.log_alpha)
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]

    plt.bar(np.arange(len(sorted_indices)), sorted_p)
    plt.plot(np.arange(len(sorted_indices)), [0.2] * len(sorted_indices), "r--")
    plt.savefig(Path(output_dir, "p_dropout.png"))


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

    sparse_active_dimensions(best_model, model_path)
