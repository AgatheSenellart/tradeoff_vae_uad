import math

import pandas as pd
import torch
from torch.nn.functional import one_hot


def bin_one_hot(data: pd.Series, num_classes: int) -> pd.Series:
    bins = pd.cut(data, bins=num_classes, labels=list(range(num_classes)))
    output = bins.apply(
        lambda x: one_hot(torch.tensor(x), num_classes=num_classes).float()
    )
    output = torch.stack(list(output.values)).float()

    return output


def conv_output_shape(size, k, s, p):
    output_size = []
    for w in size:
        output_size.append((w - k + 2 * p) // s + 1)
    return output_size


def one_hot_sex(sex: pd.Series):
    encodings = sex.map({"M": torch.tensor([0.0, 1.0]), "F": torch.tensor([1.0, 0.0])})
    encodings = torch.stack(list(encodings.values)).float()
    return encodings


def get_encoding_fct_age(args):
    if args.name == "one_hot":

        def age_encoding_fct(x):
            return bin_one_hot(x, args.nb_bins)

        return age_encoding_fct
    elif args.name == "sinusoidal":
        return SinusoidalPosEmb(args.dim, theta=args.theta)
    else:
        return NotImplementedError()


def get_encoding_fct_sex(args):
    if args.name == "one_hot":
        return one_hot_sex
    else:
        return NotImplementedError()


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, theta=10000, normalize=False, normalization_cst=100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.normalize = normalize
        self.normalization_cst = normalization_cst

    def forward(self, x: pd.Series) -> torch.Tensor:
        x = torch.from_numpy(x.values).float()
        if self.normalize:
            x = x / self.normalization_cst
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def resolve_batch_size(dim: str) -> int:
    """Resolve batch size based on the dimension of the input data."""
    if dim == "2d":
        return 2  # 64
    elif dim == "3d":
        return 1  # 4
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
