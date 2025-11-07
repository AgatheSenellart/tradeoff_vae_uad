"""To define evaluation functions"""

import logging
import os
from glob import glob
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from multivae.trainers.base.base_trainer import set_inputs_to_device
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvae_ad.model.generate_healthy import BasePredict

logger = logging.Logger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def get_anomaly_map_and_score(
    batch, reconstruction_method, brain_mask, save_dir, mean=None, std=None
):
    """Computes z, recon, anomaly map and anomaly score"""

    # Compute reconstruction
    output = reconstruction_method(batch)
    reconstruction = output["pet_linear"].detach()
    z = output["embedding"].detach() if output["embedding"] is not None else None

    if hasattr(output, "metrics"):
        save_log_prob_graphs(output.metrics, batch, save_dir)

    anomaly_map = batch["data"]["pet_linear"] - reconstruction

    if mean is not None and std is not None:
        anomaly_map = (anomaly_map - mean) / (std + 1e-8)

    anomaly_score = batch_anomaly_score_from_map(anomaly_map, brain_mask)

    return z, reconstruction, anomaly_map, anomaly_score


def batch_anomaly_score_from_map(anomaly_maps, brain_mask):
    """takes a batch tensor of anomaly maps and return the list of anomaly scores for all images in the batch"""
    anomaly_score = (
        anomaly_maps.abs()
        .reshape(len(anomaly_maps), -1)[:, brain_mask.bool().flatten()]
        .mean(-1)
    )
    assert anomaly_score.shape == (len(anomaly_maps),)
    anomaly_score = list(anomaly_score.detach().cpu().numpy())
    return anomaly_score


def get_reconstruction_metrics(input_image, reconstruction, mask):
    mse_ = batch_compute_mse(input_image, reconstruction, mask)
    ssim_ = batch_compute_ssim(input_image, reconstruction, mask)

    return mse_, ssim_


def batch_compute_ssim(batch_x, batch_x_hat, mask):
    batch_x = apply_mask_to_batch_cpu_numpy(batch_x, mask)
    batch_x_hat = apply_mask_to_batch_cpu_numpy(batch_x_hat, mask)

    results = []
    for i, x in enumerate(batch_x):
        results.append(ssim(x, batch_x_hat[i], data_range=1))

    return results


def get_eval_statistics(
    dataset,
    model,
    brain_mask,
    hypo_mask,
    save_dir,
    reconstruction_method=None,
    device="cuda",
    batch_size=2,
    save_recon=0,
    save_z=0,
    save_stats=True,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Performs full evaluation on the validation set and save the mean and std errors of the
    reconstruction to compute anomaly maps later on test datasets.
    """
    logger.info("Using method %s", reconstruction_method.__str__())
    # set the model to device
    model = model.to(device).eval()
    # set the dataloader
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the method for hypo reconstruction if not provided
    if reconstruction_method is None:
        reconstruction_method = BasePredict(model=model, use_mean_embedding=True)

    # Keep track of the anomaly maps and z
    anomaly_maps = []
    zs = []

    # Dictionary for the metrics
    results_dict = {
        "participant_id": [],
        "session_id": [],
        "MSE": [],
        "SSIM": [],
        "healthiness": [],
    }

    Path(save_dir).mkdir(exist_ok=True)

    # iterate on batchs
    for i, batch in enumerate(eval_loader):
        # set inputs to device
        batch = set_inputs_to_device(batch, device)

        z, recon, anomaly_map, _ = get_anomaly_map_and_score(
            batch, reconstruction_method, brain_mask, save_dir=save_dir
        )
        # Compute metrics
        mse, ssim_ = get_reconstruction_metrics(
            batch["data"]["pet_linear"], recon, brain_mask
        )

        results_dict["participant_id"].extend(batch["participant"])
        results_dict["session_id"].extend(batch["session"])
        results_dict["MSE"].extend(mse)
        results_dict["SSIM"].extend(ssim_)
        results_dict["healthiness"].extend(
            batch_healthiness(recon, hypo_mask, brain_mask)
        )

        if i < save_recon:
            save_tensors(batch, recon, save_dir, "_recon_")
        if i < save_z:
            if z is not None:
                save_tensors(batch, z, save_dir, "_z_")

        zs.append(z)
        anomaly_maps.append(anomaly_map)

    # Compute the mean and std of the absolute reconstruction error
    anomaly_maps = torch.cat(anomaly_maps)
    mean = torch.mean(anomaly_maps, dim=0).squeeze(0)
    std = torch.std(anomaly_maps, dim=0).squeeze(0)

    # renormalize the anomaly maps
    anomaly_maps = (anomaly_maps - mean) / std

    # Compute the anomaly score after normalization
    results_dict["anomaly_score"] = batch_anomaly_score_from_map(
        anomaly_maps, brain_mask
    )

    # Compute stats on the embeddings
    mean_z = torch.cat(zs).mean(0)
    cov_z = torch.cov(torch.cat(zs).transpose(1, 0))

    # Compute threshold on the val anomaly scores
    thresh = compute_anomaly_score_threshold(results_dict["anomaly_score"], q=0.95)

    # Save the mean and std
    if save_stats:
        torch.save(mean.half(), f"{save_dir}/mean_val.pt")
        torch.save(std.half(), f"{save_dir}/std_val.pt")
        torch.save(mean_z.half(), f"{save_dir}/mean_val_z.pt")
        torch.save(cov_z.half(), f"{save_dir}/cov_val_z.pt")

    # Save all the statistics/metrics
    df = pd.DataFrame(results_dict)
    df["threshold"] = thresh
    df["pred_label"] = (df["anomaly_score"] > thresh) * 1
    df.to_csv(f"{save_dir}/metrics_on_val.csv", float_format="{:.6f}".format)

    return mean, std, thresh, mean_z, cov_z


def get_train_z_statistics(
    dataset,
    model,
    brain_mask,
    save_dir,
    reconstruction_method=None,
    device="cuda",
    batch_size=2,
    save_recon=0,
    save_z=0,
    save_stats=True,
):
    """
    Compute the mean and std of training embeddings.
    """
    logger.info("Using method %s", reconstruction_method.__str__())
    # set the model to device
    model = model.to(device).eval()
    # set the dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs = []
    # Initialize the method for hypo reconstruction if not provided
    if reconstruction_method is None:
        reconstruction_method = BasePredict(model=model, use_mean_embedding=True)

    Path(save_dir).mkdir(exist_ok=True)

    # iterate on batchs
    for i, batch in enumerate(train_loader):
        # set inputs to device
        batch = set_inputs_to_device(batch, device)

        z, recon, _, _ = get_anomaly_map_and_score(
            batch, reconstruction_method, brain_mask, save_dir=save_dir
        )

        if i < save_recon:
            save_tensors(batch, recon, save_dir, "_recon_")
        if i < save_z:
            if z is not None:
                save_tensors(batch, z, save_dir, "_z_")

        zs.append(z)

    mean_z = torch.cat(zs).mean(0)
    cov_z = torch.cov(torch.cat(zs).transpose(1, 0))
    if save_stats:
        torch.save(mean_z.half(), f"{save_dir}/mean_train_z.pt")
        torch.save(cov_z.half(), f"{save_dir}/cov_train_z.pt")

    return mean_z, cov_z


def compute_anomaly_score_threshold(scores, q=0.95):
    return np.quantile(scores, q)


def save_tensors(batch, reconstruction, save_tensors_dir, suffix):
    """Saves the reconstructions tensors in a given_dir."""
    Path(save_tensors_dir).mkdir(exist_ok=True)
    for i, x in enumerate(reconstruction):
        torch.save(
            x,
            f"{save_tensors_dir}/{batch.participant[i]}_{batch.session[i]}_{suffix}.pt",
        )


def load_tensors(batch, save_tensors_dir, device):
    """used to load pre-computed reconstructions and embeddings."""
    recon = []
    z = []
    for i, part in enumerate(batch.participant):
        file = f"{save_tensors_dir}/{part}_{batch.session[i]}__recon_.pt"
        recon.append(torch.load(file, map_location=device))
        file_z = f"{save_tensors_dir}/{part}_{batch.session[i]}__z_.pt"
        if os.path.exists(file_z):
            z.append(torch.load(file_z, map_location=device))
        else:
            z = None

    recon = torch.stack(recon, dim=0)
    if isinstance(z, list):
        z = torch.stack(z, dim=0)
    return recon, z


def get_test_dataset_metrics(
    dataset,
    model,
    brain_mask,
    hypo_mask,
    val_dataset_stats: list,
    save_dir,
    dataset_is_simulated: bool,
    reconstruction_method=None,
    device="cuda",
    batch_size=2,
    save_recon=0,
    save_z=0,
    save_snippets=0,
    save_stats=True,
):
    """Perform full evaluation on a test dataset (simulated or not)."""

    logger.info("Using method %s", reconstruction_method.__str__())
    # set the model to device
    model = model.to(device).eval()
    # set the dataloader for iterating on the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the method for hypo reconstruction if not provided
    if reconstruction_method is None:
        reconstruction_method = BasePredict(model=model, use_mean_embedding=True)

    # keep track of the embeddings
    zs = []

    # Metrics dictionary
    metrics_dict = {
        "participant_id": [],
        "session_id": [],
        "anomaly_score": [],
        "healthiness": [],
    }

    # For non-simulated dataset, we only compare with the input
    comparisons = ["with_input"]

    # For simulated dataset, we can also compare with the ground truth
    if dataset_is_simulated:
        metrics_dict.update({"AveragePrecision": [], "BestDice": []})
        comparisons.append("with_true_healthy")

    # Keep track of reconstruction metrics
    reconstruction_metrics = {
        f"{metric}_{comp}{region}": []
        for metric in ["MSE", "SSIM"]
        for comp in comparisons
        for region in ["", "_in_mask", "_out_mask"]
    }

    # Compute metrics in the brain mask and in and out of the hypo mask
    dict_mask = {
        "": brain_mask,
        "_in_mask": hypo_mask,
        "_out_mask": brain_mask - hypo_mask,
    }
    dict_comp = {
        "with_true_healthy": lambda x: x.original_image,
        "with_input": lambda x: x.data["pet_linear"],
    }

    # Combine all metrics into a single dictionary
    metrics_dict.update(reconstruction_metrics)

    # get mean and std for the normalization of the anomaly maps
    mean, std, thresh = val_dataset_stats

    # Create the directory to save the results
    Path(save_dir).mkdir(exist_ok=True)

    # iterate on batchs
    for i, batch in enumerate(dataloader):
        # set inputs to device
        batch = set_inputs_to_device(batch, device)

        # Compute the anomaly map
        z, recon, anomaly_map, anomaly_score = get_anomaly_map_and_score(
            batch,
            reconstruction_method,
            brain_mask,
            save_dir=save_dir,
            mean=mean,
            std=std,
        )

        metrics_dict["participant_id"].extend(batch.participant)
        metrics_dict["session_id"].extend(batch.session)
        metrics_dict["anomaly_score"].extend(anomaly_score)
        metrics_dict["healthiness"].extend(
            batch_healthiness(recon, hypo_mask, brain_mask)
        )

        # get all the MSE and SSIM
        for comp in comparisons:
            image_comp = dict_comp[comp](batch)  # get the image for comparison
            for region, mask in dict_mask.items():
                # Compute reconstruction_metrics
                mse_, ssim_ = get_reconstruction_metrics(image_comp, recon, mask)

                metrics_dict[f"MSE_{comp}{region}"].extend(mse_)
                metrics_dict[f"SSIM_{comp}{region}"].extend(ssim_)

        # get the AveragePrecision and BestDice
        if dataset_is_simulated:
            aps, best_dices = batch_average_precision_and_best_dice(
                anomaly_map, brain_mask, hypo_mask
            )
            metrics_dict["AveragePrecision"].extend(aps)
            metrics_dict["BestDice"].extend(best_dices)

        if i < save_recon:
            save_tensors(batch, recon, save_dir, "_recon_")
        if i < save_z:
            if z is not None:
                save_tensors(batch, z, save_dir, "_z_")
        if i < save_snippets:
            batch_save_snippets(batch, anomaly_map, brain_mask, hypo_mask, save_dir)

        zs.append(z)

    # Create a Dataframe for the results
    df = pd.DataFrame(metrics_dict)
    # Add the predicted label
    df["pred_label"] = (df["anomaly_score"] > thresh) * 1
    df["thresh"] = thresh
    # Save the results
    df.to_csv(f"{save_dir}/metrics_on_test.csv", float_format="{:.6f}".format)

    # Save some statistics
    mean_z = torch.cat(zs).mean(0)
    cov_z = torch.cov(torch.cat(zs).transpose(1, 0))
    if save_stats:
        torch.save(mean_z.half(), f"{save_dir}/mean_test_z.pt")
        torch.save(cov_z.half(), f"{save_dir}/cov_test_z.pt")

    return mean_z, cov_z


def batch_save_snippets(batch, anomaly_maps, brain_mask, hypo_mask, save_dir):
    for j, scores in enumerate(anomaly_maps):
        save_snippet(
            scores * brain_mask,
            hypo_mask,
            Path(
                save_dir,
                "scores_snippet",
                f"participant_{batch.participant[j]}_session_{batch.session[j]}",
            ),
            largest_abs_cmap=6.5,
        )


def batch_healthiness(recon, hypo_mask, brain_mask):
    average_in_mask = recon.reshape(len(recon), -1)[:, hypo_mask.flatten().bool()].mean(
        -1
    )
    average_out_mask = recon.reshape(len(recon), -1)[
        :, (brain_mask - hypo_mask).flatten().bool()
    ].mean(-1)
    return list((average_in_mask / average_out_mask).detach().cpu().numpy())


def batch_average_precision_and_best_dice(anomaly_map, brain_mask, hypo_mask):
    aps = []
    best_dices = []

    for scores in anomaly_map:
        # Flatten the scores, hypo_mask and brain_mask
        scores = scores.detach().cpu().flatten().abs()
        labels = hypo_mask.int().flatten().cpu()
        brain_mask_flatten = brain_mask.flatten().cpu().bool()

        # Select only the voxels inside the brain mask
        scores = scores[brain_mask_flatten]
        labels = labels[brain_mask_flatten]

        # Compute average precision
        aps.append(average_precision_score(labels, scores.numpy()))

        # Compute BestDice
        best_dices.append(best_dice(labels, scores)[0])

    return aps, best_dices


def save_snippet(scores, hypo_mask, output_path, largest_abs_cmap=None):
    """Save a snippet of the score to png format"""
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    os.makedirs(output_path, exist_ok=True)

    if scores.shape == (1, 169, 208, 179):
        slices = [
            (
                scores[0, 80, :, :].cpu().numpy(),
                hypo_mask[80, :, :].cpu().numpy(),
                "new_scores_axis_0.png",
            ),
            (
                scores[0, :, 104, :].cpu().numpy(),
                hypo_mask[:, 104, :].cpu().numpy(),
                "new_scores_axis_1.png",
            ),
            (
                scores[0, :, :, 80].cpu().numpy(),
                hypo_mask[:, :, 80].cpu().numpy(),
                "new_scores_axis_2.png",
            ),
        ]

        for score_slice, mask_slice, filename in slices:
            fig, axes = plt.subplots(1, 3, figsize=(8, 4))

            cmap = "seismic"
            qmin = np.quantile(score_slice, 0.003)
            qmax = np.quantile(score_slice, 0.997)
            largest_abs = max(abs(qmin), abs(qmax))
            norm = MidpointNormalize(midpoint=0, vmin=-largest_abs, vmax=largest_abs)

            axes[0].imshow(np.rot90(score_slice), cmap=cmap, norm=norm)
            axes[0].set_title("Scores")
            axes[0].axis("off")

            axes[1].imshow(
                np.rot90(np.abs(score_slice)),
                cmap="Greys",
                norm=matplotlib.colors.Normalize(vmin=0, vmax=largest_abs * 0.75),
            )
            axes[1].set_title("Absolute value of scores")
            axes[1].axis("off")

            axes[2].imshow(np.rot90(mask_slice), cmap="Greys", norm=None)
            axes[2].set_title("Hypo Mask")
            axes[2].axis("off")

            for ax in axes[:2]:
                im = ax.images[0]
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, filename))
            plt.close(fig)

    elif scores.shape == (1, 169, 208):
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

        cmap = "seismic"
        qmin = np.quantile(scores.cpu().numpy(), 0.003)
        qmax = np.quantile(scores.cpu().numpy(), 0.997)

        if largest_abs_cmap is not None:
            largest_abs = largest_abs_cmap
        else:
            largest_abs = max(abs(qmin), abs(qmax))
        norm = MidpointNormalize(midpoint=0, vmin=-largest_abs, vmax=largest_abs)

        axes[0].imshow(
            np.rot90(scores[0, :, :].cpu().numpy()), cmap="seismic", norm=norm
        )
        axes[0].set_title("Scores")
        axes[0].axis("off")
        axes[1].imshow(
            np.rot90(np.abs(scores[0, :, :].cpu().numpy())),
            cmap="Greys",
            norm=matplotlib.colors.Normalize(vmin=0, vmax=largest_abs * 0.75),
        )
        axes[1].set_title("Absolute value of scores")
        axes[1].axis("off")
        axes[2].imshow(np.rot90(hypo_mask.cpu().numpy()), cmap="Greys", norm=None)
        axes[2].set_title("Hypo Mask")
        axes[2].axis("off")
        for ax in axes[:2]:
            im = ax.images[0]
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "new_scores.png"))
        plt.close(fig)

    else:
        raise ValueError("The scores shape is not supported for saving snippets.")


def save_reconstruction_fct(batch, xprime, xhat, xprimehat, output_dir):
    # Set reconstruction_caps_dir
    reconstruction_caps_dir = os.path.join(
        output_dir,
        "reconstruction_caps",
    )
    os.makedirs(reconstruction_caps_dir, exist_ok=True)

    x = batch["data"]["pet_linear"]

    # Input
    for i, t in enumerate(x):
        t_nii = nib.Nifti1Image(t[0].detach().cpu().numpy(), np.eye(4))
        filename_nii = f"{batch.participant[i]}_{batch.session[i]}_input.nii.gz"
        nib.save(t_nii, os.path.join(reconstruction_caps_dir, filename_nii))

    # Input hypo
    for i, t in enumerate(xprime):
        t_nii = nib.Nifti1Image(t[0].detach().cpu().numpy(), np.eye(4))
        filename_nii = f"{batch.participant[i]}_{batch.session[i]}_input_hypo.nii.gz"
        nib.save(t_nii, os.path.join(reconstruction_caps_dir, filename_nii))

    # Reconstruction
    for i, t in enumerate(xhat):
        t_nii = nib.Nifti1Image(t[0].detach().cpu().numpy(), np.eye(4))
        filename_nii = f"{batch.participant[i]}_{batch.session[i]}_output.nii.gz"
        nib.save(t_nii, os.path.join(reconstruction_caps_dir, filename_nii))

    # Reconstruction hypo
    for i, t in enumerate(xprimehat):
        t_nii = nib.Nifti1Image(t[0].detach().cpu().numpy(), np.eye(4))
        filename_nii = f"{batch.participant[i]}_{batch.session[i]}_output_hypo.nii.gz"
        nib.save(t_nii, os.path.join(reconstruction_caps_dir, filename_nii))


def get_hydra_config_and_model(hydra_path, window=None) -> Tuple[DictConfig, str]:
    """
    In this function, we reload the hydra configuration of a previous path,
    we reload the model and the ml_flow id for futher evaluation of the
    trained model.
    """

    # get hydra configuration
    cfg = OmegaConf.load(os.path.join(hydra_path, ".hydra", "config.yaml"))

    # get the model_path
    if window is None:
        # If no window is specified, we load the final model
        model_path = glob(os.path.join(hydra_path, "CVAE*", "final_model"))[0]

    else:
        model_path = glob(
            os.path.join(
                hydra_path,
                "CVAE*",
                "best_models_per_window",
                f"checkpoint_epoch_{window}",
            )
        )[0]

    return cfg, model_path


def apply_mask_to_batch(tensors, mask):
    """tensors :  (bs, w,l,h),
    mask = (w,l,h)"""

    return tensors * mask


def apply_mask_to_batch_cpu_numpy(batch, mask):
    return apply_mask_to_batch(batch, mask).cpu().numpy().squeeze(1)


def apply_hypo_mask_in_out(img, brain_mask, hypo_mask):
    in_hypo = apply_mask_to_batch_cpu_numpy(img, hypo_mask)
    out_hypo = apply_mask_to_batch_cpu_numpy(img, brain_mask - hypo_mask)
    return in_hypo, out_hypo


def batch_compute_mse(img1, img2, mask):
    img1 = apply_mask_to_batch_cpu_numpy(img1, mask)
    img2 = apply_mask_to_batch_cpu_numpy(img2, mask)
    return list(((img1 - img2) ** 2).reshape(len(img1), -1).mean(axis=1))


def reconstruction_error_depending_on_pixel_values(
    eval_dataset,
    model,
    output_dir,
    device="cuda",
    n_images=10,
):
    """
    Compute the reconstruction error depending on the pixel values.
    """

    model = model.to(device)

    # create bins between 0 and 1 for pixel values
    bins = np.linspace(0, 1, 11)

    list_df = []
    # iterate on the images
    for i in range(n_images):
        sample = eval_dataset[i]
        sample["data"]["pet_linear"] = (
            sample["data"]["pet_linear"].to(device).unsqueeze(0)
        )

        # Compute reconstruction
        reconstruction = model.predict(sample, use_mean_embedding=True)["pet_linear"]

        # Compute reconstruction error
        reconstruction_error = torch.abs(reconstruction - sample["data"]["pet_linear"])

        # For each pixel, compute the bin it belongs to
        pixel_values = sample["data"]["pet_linear"].cpu().numpy().flatten()
        pixel_bins = np.digitize(pixel_values, bins) - 1
        pixel_bins = pixel_bins.reshape(sample["data"]["pet_linear"].shape)
        pixel_bins = torch.tensor(pixel_bins, device=device)
        # Compute the mean reconstruction error for each bin
        errors_per_bin_mean = []
        for bin_idx in range(len(bins) - 1):
            mask = pixel_bins == bin_idx
            errors_per_bin_mean.append(reconstruction_error[mask].mean().item())

        # Save the results to a csv file
        results_df = pd.DataFrame({"bin": bins[:-1], "mean_error": errors_per_bin_mean})
        results_df["participant"] = sample["participant"]
        results_df["session"] = sample["session"]
        results_df["age_float"] = sample["age_float"]

        list_df.append(results_df)

    # Concatenate all the results
    final_df = pd.concat(list_df, ignore_index=True)
    # Save the final dataframe to a csv file
    final_df.to_csv(
        os.path.join(output_dir, "reconstruction_error_per_pixel.csv"), index=False
    )
    return final_df


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0,
            1
            / 2
            * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
        )
        normalized_mid = 0.5
        x, y = (
            [self.vmin, self.midpoint, self.vmax],
            [
                normalized_min,
                normalized_mid,
                normalized_max,
            ],
        )
        return np.ma.masked_array(np.interp(value, x, y))


def save_log_prob_graphs(metrics, batch, output_dir):
    """For reconstruction methods that include optimization or hmc sampling, save the evolution of losses."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    for i, (participant, session) in enumerate(
        zip(batch["participant"], batch["session"])
    ):
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Itérations")
        ax1.set_ylabel("loss", color=color)
        ax1.plot(metrics["losses"][i].cpu(), color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        if "prior" in metrics.keys():
            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

            color = "tab:blue"
            ax2.set_ylabel(
                "prior", color=color
            )  # we already handled the x-label with ax1
            ax2.plot(metrics["prior"][i].cpu(), color=color)
            ax2.tick_params(axis="y", labelcolor=color)

        if "recon" in metrics.keys():
            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

            color = "tab:green"
            ax2.set_ylabel(
                "recon", color=color
            )  # we already handled the x-label with ax1
            ax2.plot(metrics["recon"][i].cpu(), color=color)
            ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        Path(output_dir, "losses_hypo_reconstruction").mkdir(exist_ok=True)
        plt.savefig(
            Path(
                output_dir,
                "losses_hypo_reconstruction",
                f"losses_{participant}_{session}.png",
            )
        )


def compute_dice(flatten_preds, flatten_targets) -> float:
    """Compute the DICE score. This only works for segmentations.
    PREDICTIONS NEED TO BE BINARY!

    Args:
        predictions (torch.tensor): Predicted binary anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
    Returns:
        dice (float)
    """

    pred_sum = flatten_preds.sum()
    targ_sum = flatten_targets.sum()
    intersection = flatten_preds.float() @ flatten_targets.float()
    dice = (2 * intersection) / (pred_sum + targ_sum)
    return dice


def best_dice(targets: torch.Tensor, preds: torch.Tensor, n_thresh=100):
    """For one image, takes the ground truth mask and the scores and compute the best dice score.
    WARNING: for now we don't implement the connected components procedure.
    """

    thresholds = np.linspace(preds.min(), preds.max(), n_thresh)
    threshs = []
    scores = []
    pbar = tqdm(thresholds, desc="DICE search")
    for t in pbar:
        dice = compute_dice(torch.where(preds > t, 1.0, 0.0), targets)
        scores.append(dice)
        threshs.append(t)

    scores = torch.stack(scores, 0)
    max_dice = scores.max().item()
    max_thresh = threshs[scores.argmax()]

    # Get best dice once again after connected component analysis
    # bin_preds = torch.where(preds > max_thresh, 1., 0.)
    # # bin_preds = connected_components_3d(bin_preds)
    # max_dice = compute_dice(bin_preds, targets)
    return max_dice, max_thresh
