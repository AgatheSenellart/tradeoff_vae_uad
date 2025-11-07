import logging
import os

import nibabel as nib
import numpy as np
import torch
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.base_trainer import set_inputs_to_device
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid, save_image

from mvae_ad.data.caps_multimodal_dataset import CapsMultimodalDataset

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


@dataclass
class CapsTrainerConfig(BaseTrainerConfig):
    """Configuration for CapsTrainer, inheriting from BaseTrainerConfig.
    Is used to specify additional parameters.
    - window_size_for_checkpoints: int, the number of epochs in each window for which we save the best model.
    - start_keep_best_epoch: int, the epoch from which we start keeping the best model overall.

    If you don't want to use the window behavior, you can set 'window_size_for_checkpoints'
        to the max_number of epochs.


    """

    start_keep_best_epoch: int = 10
    save_nifti_reconstruction: bool = False


class CapsTrainer(BaseTrainer):
    """Changes the predict function of BaseTrainer to save reconstructions
    in a CAPS format.
    We can put anything else we want in this function for other analysis
    we might want to do during training.
    """

    def __init__(
        self,
        model,
        train_dataset: CapsMultimodalDataset,
        eval_dataset: CapsMultimodalDataset,
        training_config: CapsTrainerConfig,  # just to be sure we use the right config
        callbacks=None,
        checkpoint=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            callbacks=callbacks,
            checkpoint=checkpoint,
        )

        assert training_config.start_keep_best_epoch < training_config.num_epochs, (
            "The start keep best epoch must be less than the number of epochs."
        )
        self.start_keep_best_epoch = training_config.start_keep_best_epoch

    def predict(self, model, epoch, n_data=1, save_reconstruction=False):
        """Compute reconstructions for the entire eval set and saves them in a Caps format"""
        with torch.no_grad():
            model.eval()

            # Directory for saving the reconstruction caps
            recon_caps_dir = os.path.join(self.training_dir, "reconstruction_caps")
            os.makedirs(recon_caps_dir, exist_ok=True)

            # Create output dictionary to save everything
            output = {}

            predict_loaders = {"train": self.train_loader, "eval": self.eval_loader}

            ### Compute reconstructions and metrics on the eval dataset
            for split, predict_loader in predict_loaders.items():
                torch.cuda.empty_cache()
                if predict_loader is not None:
                    metrics = [
                        MeanAbsoluteError().to(self.device),
                        MeanSquaredError().to(self.device),
                        PeakSignalNoiseRatio(data_range=1).to(self.device),
                        StructuralSimilarityIndexMeasure().to(self.device),
                        #    MultiScaleStructuralSimilarityIndexMeasure().to(self.device)
                    ]
                    for batch in predict_loader:
                        batch = set_inputs_to_device(batch, self.device)

                        # get the image reconstruction
                        output_batch = model.predict(batch)
                        z = output_batch["embedding"]  # n_batch, z_dim
                        reconstructions = output_batch[
                            "pet_linear"
                        ]  # n_batch, *data_dim

                        # compute metrics
                        for m in metrics:
                            m(reconstructions, batch["data"]["pet_linear"])

                        # Save the reconstructed tensors
                        if save_reconstruction:
                            for i, t in enumerate(reconstructions):
                                # make the dir
                                sample_dir = os.path.join(
                                    recon_caps_dir,
                                    batch.participant[i],
                                    batch.session[i],
                                    "pet_linear",
                                    "tensors",
                                )
                                os.makedirs(sample_dir, exist_ok=True)
                                filename = "reconstruction.pt"
                                # Save the reconstructed tensor
                                torch.save(t, os.path.join(sample_dir, filename))

                    # Aggregate the metrics and log them
                    for m in metrics:
                        output[f"{m._get_name()}_on_{split}"] = m.compute().item()

            torch.cuda.empty_cache()
            # Save a snippet of a few samples to easily visualize if training is going well or not
            sample = next(iter(DataLoader(self.train_dataset, n_data)))
            sample = set_inputs_to_device(sample, self.device)
            reconstruction = model.predict(sample)["pet_linear"]  # (nb_batch, ch, w, h)
            truth = sample.data["pet_linear"]

            grid_images = self.save_snippet(
                true_image=truth,
                reconstruction=reconstruction,
                n_data=n_data,
                recon_caps_dir=recon_caps_dir,
                epoch=str(epoch),
            )
            # Save the nifty files
            if self.training_config.save_nifti_reconstruction:
                self.save_nifti(
                    true_image=truth,
                    reconstruction=reconstruction,
                    n_data=n_data,
                    recon_caps_dir=recon_caps_dir,
                    epoch=str(epoch),
                )
            # Add the snippet images to the output
            output["images"] = grid_images

            return output

    def save_nifti(self, true_image, reconstruction, n_data, recon_caps_dir, epoch=""):
        if true_image.shape[1:] == (1, 169, 208, 179):
            nib_reconstruction = nib.Nifti1Image(
                reconstruction[0, 0].cpu().numpy(), affine=np.eye(4)
            )

            nib_true_image = nib.Nifti1Image(
                true_image[0, 0].cpu().numpy(), affine=np.eye(4)
            )

            # Save the true image
            nib.save(
                nib_true_image,
                os.path.join(recon_caps_dir, f"true_image_epoch_{epoch}.nii.gz"),
            )
            # Save the reconstruction

            nib.save(
                nib_reconstruction,
                os.path.join(recon_caps_dir, f"reconstruction_epoch_{epoch}.nii.gz"),
            )

    def save_snippet(
        self, true_image, reconstruction, n_data, recon_caps_dir, epoch=""
    ):
        # For 2D images, we just save the image and its reconstruction

        if true_image.shape[1:] == (1, 169, 208):
            concat = torch.cat([true_image, reconstruction])
            grid_images = {"snippet": make_grid(concat, n_data)}

        # For 3D images, we select the middle slice in each direction
        elif true_image.shape[1:] == (1, 169, 208, 179):
            concat_1 = torch.cat(
                [true_image[:, :, 80, :, :], reconstruction[:, :, 80, :, :]]
            )
            concat_2 = torch.cat(
                [true_image[:, :, :, 100, :], reconstruction[:, :, :, 100, :]]
            )
            concat_3 = torch.cat(
                [true_image[:, :, :, :, 80], reconstruction[:, :, :, :, 80]]
            )
            grid_images = {
                "snippet_1": make_grid(concat_1, n_data),
                "snippet_2": make_grid(concat_2, n_data),
                "snippet_3": make_grid(concat_3, n_data),
            }

        else:
            raise AttributeError("The image is not 2D or 3D")

        snippet_path = os.path.join(recon_caps_dir, "snippet")
        os.makedirs(snippet_path, exist_ok=True)

        # Save the image and add it to the output
        for k, image in grid_images.items():
            save_image(image, os.path.join(snippet_path, f"epoch_{epoch}_{k}.png"))

        return grid_images
