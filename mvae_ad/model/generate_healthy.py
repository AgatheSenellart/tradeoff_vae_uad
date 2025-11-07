"""In this module we implement different strategies to generate the pseudo healthy image
using the VAE and a model of the healthy distribution in the latent space"""

import logging

import torch
from multivae.data.utils import DatasetOutput
from multivae.models.base import ModelOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# add ch to logger
logger.addHandler(ch)


class BasePredict(object):
    """
    Generate a pseudo healthy image from the latent space using a simple encode and decode.

    Args:
        x (torch.tensor): The input tensor to encode.

    Returns:
        torch.tensor: The generated pseudo healthy image.
    """

    def __init__(self, model, use_mean_embedding=True, **kwargs):
        self.model = model.eval()
        self.use_mean_embedding = use_mean_embedding

    def __call__(self, x: DatasetOutput, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self.model.predict(x, use_mean_embedding=self.use_mean_embedding)


class MeanPredict(BasePredict):
    """Take the mean of the training embeddings to reconstruct all
    images."""

    def __init__(self, model, train_embeddings, **kwargs):
        self.model = model
        self.mean_train_embeddings = train_embeddings.mean(0)

    def __call__(self, x, **kwargs):
        with torch.no_grad():
            output_encode = self.model.encode(x)
            cond_mod_data = output_encode.cond_mod_data
            z = torch.stack(
                [self.mean_train_embeddings] * x.data["pet_linear"].shape[0]
            )
            recon = self.model.decode(
                ModelOutput(z=z, cond_mod_data=cond_mod_data)
            ).reconstruction

        return ModelOutput(embedding=z, pet_linear=recon)
