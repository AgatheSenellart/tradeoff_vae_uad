from copy import deepcopy
from pathlib import Path

import numpy as np
import torchio as tio
from clinicadl.data.structures import DataPoint


class SimulateHypometabolic(tio.Transform):
    """To simulate dementia-related hypometabolism
    Ref. https://www.melba-journal.org/papers/2024:003.html."""

    def __init__(
        self,
        mask_dir: str | Path,
        pathology: str,
        percentage: int,
        sigma: int = 2,
        **kwargs,
    ):
        import nibabel as nib

        super().__init__(**kwargs)

        self.pathology = pathology
        self.percentage = percentage
        self.sigma = sigma
        self.mask_dir = mask_dir

        mask_path = Path(
            self.mask_dir, f"mask_hypo_{self.pathology.lower()}_resampled.nii"
        )
        mask_nii = nib.load(mask_path)
        self.mask = self._mask_processing(mask_nii.get_fdata())

        self.args_names = ["mask_dir", "pathology", "percentage", "sigma"]

    def apply_transform(
        self,
        datapoint: DataPoint,
    ) -> DataPoint:
        transformed = deepcopy(datapoint)

        for image in transformed.get_images(intensity_only=True):
            image.tensor[:] *= self.mask

        transformed.add_image(datapoint.image, "original_image")
        transformed.add_mask(
            tio.LabelMap(
                tensor=np.expand_dims(self.mask, axis=0), affine=datapoint.image.affine
            ),
            "hypo_mask",
        )
        transformed["pathology"] = self.pathology
        transformed["percentage"] = self.percentage

        return transformed

    def _mask_processing(self, mask: np.array) -> np.array:
        from scipy.ndimage import gaussian_filter

        inverse_mask = 1 - mask
        inverse_mask[inverse_mask == 0] = 1 - self.percentage / 100
        gaussian_mask = gaussian_filter(inverse_mask, sigma=self.sigma)
        return np.float32(gaussian_mask)
