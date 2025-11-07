import logging
import os

import pandas as pd
import torch

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)


def grid_search_mahalanobis_distance(
    train_embeddings,
    hypo_embeddings_dict,
    output_dir,
    cov_estimation_methods_to_test=("MinCovDet", "EmpiricalCovariance"),
    already_computed_covs=None,
):
    """Compute the mahalanobis distance for the hypo_embeddings with different methods for covariance estimation."""

    covs_estimators = {}

    with torch.no_grad():
        # Initialize the metrics dict
        metrics = {
            "participant": hypo_embeddings_dict["participant"],
            "session": hypo_embeddings_dict["session"],
        }

        # Iterate over the number of components to test
        for method in cov_estimation_methods_to_test:
            logger.info(f"Testing Mahalanobis with {method} covariance estimation")
            if already_computed_covs is not None:
                score = already_computed_covs[f"mahalanobis_{method}"]
                logger.info("Using pre-computed covs")
            else:
                # Initialize the score
                score = MahalanobisDistance(covariance_estimator=method)
                score.fit(train_embeddings)
                logger.info("Computed covs")
            scores = score.compute(hypo_embeddings_dict["embeddings"].cpu())
            metrics[f"mahalanobis_{method}"] = scores
            covs_estimators[f"mahalanobis_{method}"] = score

        # Convert to dataframe and save to disk
        metrics_df = pd.DataFrame(metrics)
        output_file = os.path.join(output_dir, "mahalanobis_scores.csv")
        metrics_df.to_csv(output_file, index=False)
        return covs_estimators
