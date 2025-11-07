import torch
from multivae.models.base import ModelOutput
from multivae.models.nn.base_architectures import BaseJointEncoder
from torch import nn
from torchvision.ops import MLP

from mvae_ad.architectures.architectures_2d import get_activation_fct


class JointEncoderNetwork(nn.Module):
    """Simple network to join image information and covariate information in the encoder.

    TODO: Maybe add a cross attention layer instead of simple concatenation ?

    """

    def __init__(
        self,
        dim_image,
        dim_covars,
        hidden_dims=[128],
        aggregation="concat",
        norm_layer=None,
        activation="relu",
    ):
        super().__init__()

        self.aggregation = aggregation
        activation = get_activation_fct(activation)

        if self.aggregation != "concat":
            raise NotImplementedError(
                "Only aggregation == 'concat' is implemented for now"
            )
        self.output_dim = hidden_dims[-1]
        self.network = MLP(
            dim_covars + dim_image,
            hidden_dims,
            norm_layer=norm_layer,
            activation_layer=activation,
        )

    def forward(self, h, covars):
        if self.aggregation == "concat":
            concat = torch.cat([h, covars], dim=-1)
            return self.network(concat)

        return


class CovariatesNetwork(nn.Module):
    """Simple network to process the covariates before merging with other components.

    TODO: Maybe add an embedding layer ?

    """

    def __init__(
        self,
        cond_mods,
        input_dim,
        hidden_dims=[20, 20],
        norm_layer=None,
        activation="relu",
    ):
        super().__init__()
        activation = get_activation_fct(activation)
        if hidden_dims == []:
            self.output_dim = input_dim
            self.network = lambda x: x
        else:
            self.output_dim = hidden_dims[-1]
            self.network = MLP(
                input_dim, hidden_dims, norm_layer, activation_layer=activation
            )
        self.cond_mods = cond_mods

    def forward(self, x):
        covars = [x[mod] for mod in self.cond_mods]
        covars = torch.cat(covars, -1)
        return self.network(covars)


class PriorCovariatesNetwork(BaseJointEncoder):
    """Simple Encoder that returns the mean and the log_covariance
    of the prior when conditioning on the covariates."""

    def __init__(
        self,
        cond_mods,
        input_dim,
        latent_dim,
        hidden_dims=[20, 20],
        norm_layer=None,
        activation="relu",
    ):
        super().__init__()
        self.cond_mods = cond_mods
        self.latent_dim = latent_dim
        activation = get_activation_fct(activation)
        self.network = MLP(input_dim, hidden_dims, norm_layer, activation)
        # Since there is no activation at the end of the MLP, we add one before last linear layer
        self.mu = nn.Sequential(activation(), nn.Linear(hidden_dims[-1], latent_dim))
        self.lv = nn.Sequential(activation(), nn.Linear(hidden_dims[-1], latent_dim))

    def forward(self, x):
        covars = [x[mod] for mod in self.cond_mods]
        covars = torch.cat(covars, -1)
        h = self.network(covars)

        return ModelOutput(embedding=self.mu(h), log_covariance=self.lv(h))
