"""This file contains utilities functions for comparing latent distributions"""

import warnings

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from sklearn import covariance
from torch import nn


def wasserstein_between_gaussians(mu1, mu2, cov1, cov2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)

    if not np.isfinite(covmean).all():
        msg = (
            "wasserstein calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1):
            m = np.max(np.abs(covmean.imag))
            warnings.warn(
                "Possible micomputation !!! : Imaginary component {}".format(m)
            )
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean


def wasserstein_between_gaussians_samples(
    x1, x2, eps=1e-6, covariance_estimator="EmpiricalCovariance"
):
    """Computes the 2-wasserstein distance between distributions"""

    cov1 = getattr(covariance, covariance_estimator)()
    cov1.fit(x1)

    cov2 = getattr(covariance, covariance_estimator)()
    cov2.fit(x2)
    mu_1, cov_1 = cov1.location_, cov1.covariance_
    mu_2, cov_2 = cov2.location_, cov2.covariance_
    return wasserstein_between_gaussians(mu_1, mu_2, cov_1, cov_2, eps)


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        kernels = []
        for m in self.bandwidth_multipliers:
            kernels.append(
                torch.exp(-L2_distances / (self.get_bandwidth(L2_distances) * m))
            )
        return torch.stack(kernels).sum(0)
        # return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    """This is the biased but positive estimation of MMD ^ 2"""

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def compute_all_wasserstein_dists(z_stats: dict, save_dir):
    wasserstein_dict = {"comparison": [], "value": []}
    for name1, stats_1 in z_stats.items():
        for name2, stats_2 in z_stats.items():
            if name1 != name2:
                v = wasserstein_between_gaussians(
                    stats_1[0].detach().cpu().numpy(),
                    stats_2[0].detach().cpu().numpy(),
                    stats_1[1].detach().cpu().numpy(),
                    stats_1[1].detach().cpu().numpy(),
                )
                wasserstein_dict["comparison"].append(f"{name1}_{name2}")
                wasserstein_dict["value"].append(v)

    df = pd.DataFrame(wasserstein_dict)
    df.to_csv(f"{save_dir}/wasserstein_dist.csv", float_format="{:.6f}".format)
    return


# # For local tests
# if __name__ == '__main__':

#     # create and compare gaussians distributions
#     from torch.distributions import Normal

#     x1 = Normal(2,3).sample([100,64])
#     x2 = Normal(0,5).sample([100,64])
#     x3 = Normal(2,5).sample([100,64])
#     x4 = Normal(1.5,3).sample([100,64])

#     print(wasserstein_between_gaussians_samples(x1.numpy(),x2.numpy()))
#     print(wasserstein_between_gaussians_samples(x1.numpy(),x1.numpy()))
#     print(wasserstein_between_gaussians_samples(x1.numpy(),x3.numpy()))
#     print(wasserstein_between_gaussians_samples(x1.numpy(),x4.numpy()))


#     mmd = MMDLoss()

#     print(mmd(x1,x2).item())
#     print(mmd(x1,x1).item())
#     print(mmd(x1,x3).item())
#     print(mmd(x1,x4).item())
