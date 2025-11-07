"This file is meant to contains the resnets architectures"

import math

import torch
from multivae.models.base import ModelOutput
from multivae.models.nn.base_architectures import (
    BaseConditionalDecoder,
    BaseJointEncoder,
)
from torch import nn


def get_activation_fct(str_name):
    if str_name == "relu":
        return nn.ReLU
    elif str_name == "silu":
        return nn.SiLU
    else:
        raise NotImplementedError(
            f"The provided activation name {str_name} is not recognized. ",
            'Options are "relu" or "silu"',
        )


class ResBlock(nn.Module):
    def __init__(
        self, input_channels, output_channels, keep_dim=False, activation="relu"
    ):
        super().__init__()

        self.keep_dim = keep_dim
        self.layers = []
        activation = get_activation_fct(activation)

        if keep_dim:
            self.layers.append(
                nn.Conv2d(
                    input_channels, output_channels, kernel_size=3, stride=1, padding=1
                )
            )
        else:
            self.layers.append(
                nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1)
            )

        self.layers.extend(
            [
                nn.BatchNorm2d(output_channels),
                activation(),
                nn.Conv2d(output_channels, output_channels, 3, 1, 1),
                nn.BatchNorm2d(output_channels),
            ]
        )
        self.network = nn.Sequential(*self.layers)

        if not keep_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(output_channels),
            )
        elif input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = None

        self.output = activation()

    def forward(self, x):
        h = self.network(x)
        if self.shortcut is None:
            hx = x
        else:
            hx = self.shortcut(x)
        return self.output(h + hx)


class ConvBlock(nn.Module):
    def __init__(
        self, input_channels, output_channels, keep_dim=False, activation="relu"
    ):
        super().__init__()

        self.keep_dim = keep_dim
        self.layers = []
        activation = get_activation_fct(activation)

        if keep_dim:
            self.layers.append(
                nn.Conv2d(
                    input_channels, output_channels, kernel_size=3, stride=1, padding=1
                )
            )
        else:
            self.layers.append(
                nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1)
            )

        self.layers.extend(
            [
                nn.BatchNorm2d(output_channels),
                activation(),
            ]
        )
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        h = self.network(x)

        return h


class Encoder2d(BaseJointEncoder):
    def __init__(
        self,
        latent_dim,
        n_blocks=5,
        n_subblocks=1,
        activation="relu",
        block="conv",
        cond_mod_network=None,
        joint_network=None,
        **kwargs,
    ):
        super().__init__()

        if block == "conv":
            block = ConvBlock
        elif block == "res":
            block = ResBlock
        else:
            raise AttributeError(f"block type {block} not supported.")

        self.latent_dim = latent_dim
        self.use_cond_mods = cond_mod_network is not None

        self.layers = []

        in_channels = 1
        out_channels = 32
        image_shape = [169, 208]
        for _ in range(n_blocks):
            block_layers = [block(in_channels, out_channels, False, activation)]
            for _ in range(n_subblocks - 1):
                block_layers.append(block(out_channels, out_channels, True, activation))
            self.layers.extend(block_layers)
            in_channels = out_channels
            out_channels = out_channels * 2
            image_shape[0] = image_shape[0] // 2
            image_shape[1] = image_shape[1] // 2

        self.layers.append(nn.Flatten())
        n_features = in_channels * image_shape[0] * image_shape[1]

        ## if we use the covariates
        if self.use_cond_mods:
            self.layers.append(nn.Linear(n_features, self.latent_dim))
            self.cond_mod_encoder = cond_mod_network
            self.joint_network = joint_network
            n_features = joint_network.output_dim

        self.image_network = nn.Sequential(*self.layers)

        self.mu = nn.Linear(n_features, self.latent_dim)
        self.lv = nn.Linear(n_features, self.latent_dim)

    def forward(self, x):
        x_pet = x["pet_linear"]

        h = self.image_network(x_pet)

        if self.use_cond_mods:
            covariates = self.cond_mod_encoder(x)
            h = self.joint_network(h, covariates)

        return ModelOutput(embedding=self.mu(h), log_covariance=self.lv(h))


class Decoder2d(BaseConditionalDecoder):
    """
    Resnet Decoder.


    """

    def __init__(
        self,
        latent_dim,
        cond_mod_network=None,
        n_blocks=5,
        n_subblocks=1,
        activation="relu",
        block="conv",
        **kwargs,
    ):
        super().__init__()

        if block == "conv":
            block = ConvBlock
        elif block == "res":
            block = ResBlock
        else:
            raise AttributeError(f"block type {block} not supported.")

        self.latent_dim = latent_dim
        self.use_cond_mods = cond_mod_network is not None
        self.cond_mod_network = cond_mod_network
        activation_fct = get_activation_fct(activation)

        dense_layer_dim = latent_dim
        if self.use_cond_mods:
            dense_layer_dim += cond_mod_network.output_dim

        images_shape = [[169, 208]]
        for _ in range(n_blocks):
            new_shape = [images_shape[-1][0] // 2, images_shape[-1][1] // 2]
            images_shape.append(new_shape)

        images_shape.reverse()

        deep_shape = (32 * 2 ** (n_blocks - 1), images_shape[0][0], images_shape[0][1])

        self.layers = [
            nn.Linear(dense_layer_dim, math.prod(deep_shape)),
            activation_fct(),
            nn.Unflatten(1, unflattened_size=deep_shape),
        ]

        in_channels = deep_shape[0]
        out_channels = in_channels // 2
        for i in range(n_blocks - 1):
            blocks_layers = []
            for _ in range(n_subblocks - 1):
                blocks_layers.append(block(in_channels, in_channels, True, activation))
            blocks_layers.append(nn.Upsample(size=images_shape[i + 1]))
            # print(in_channels, out_channels)
            blocks_layers.append(block(in_channels, out_channels, True, activation))
            self.layers.extend(blocks_layers)

            in_channels = out_channels
            out_channels = out_channels // 2

        # Last block
        blocks_layers = []
        for _ in range(n_subblocks - 1):
            blocks_layers.append(block(in_channels, in_channels, True, activation))
        blocks_layers.append(nn.Upsample(size=images_shape[-1]))
        blocks_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
        blocks_layers.append(nn.BatchNorm2d(in_channels))
        blocks_layers.append(activation_fct())
        blocks_layers.append(nn.Conv2d(in_channels, 1, 3, 1, 1))
        blocks_layers.append(nn.Sigmoid())

        self.layers.extend(blocks_layers)

        self.network = nn.Sequential(*self.layers)

    def forward(self, z, cond_mods):
        # concatenate the latent dim and the conditioning modalities
        if self.use_cond_mods:
            covars = self.cond_mod_network(cond_mods)
            concat = torch.cat([z, covars], dim=-1)
        else:
            concat = z

        reconstruction = self.network(concat)

        return ModelOutput(reconstruction=reconstruction)
