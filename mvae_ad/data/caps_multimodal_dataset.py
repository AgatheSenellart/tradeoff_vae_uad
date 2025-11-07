from typing import Callable, Dict, Optional

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes.preprocessing import Preprocessing
from clinicadl.transforms import Transforms
from clinicadl.utils.typing import DataType, PathType
from multivae.data.datasets.base import DatasetOutput


class CapsMultimodalDataset(CapsDataset):
    def __init__(
        self,
        caps_directory: PathType,
        preprocessing: Preprocessing,
        transforms: Transforms,
        json_name: str,
        data: Optional[DataType] = None,
        label: Optional[str] = None,
        masks: Optional[list[PathType]] = None,
        txt_modalities: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(
            caps_directory=caps_directory,
            preprocessing=preprocessing,
            data=data,
            label=label,
            transforms=transforms,
            masks=masks,
        )

        self.read_tensor_conversion(json_name, check_transforms=True)

        self.txt_modalities = txt_modalities
        self.dim_dict = {}
        if self.txt_modalities is not None:
            self.covariates_dict = {}  # Dictionary to store the covariates after transform
            for txt_modality, encoding in self.txt_modalities.items():
                if txt_modality not in self.df.columns:
                    raise ValueError(
                        f"txt_modality {txt_modality} not found in dataframe columns"
                    )
                self.covariates_dict[txt_modality] = encoding(self.df[txt_modality])
                self.dim_dict[txt_modality] = self.covariates_dict[txt_modality].shape[
                    1:
                ]

    def __getitem__(self, index):
        X = super().__getitem__(index)

        # à adapter pour les différentes modalités
        X_dict = {}
        try:
            X_dict["pet_linear"] = X.image.data.squeeze(-1)
        except AttributeError:
            X_dict["pet_linear"] = X.get_tensors()["image"].squeeze(-1)

        if self.txt_modalities is not None:
            for txt_modality in self.txt_modalities:
                X_dict[txt_modality] = self.covariates_dict[txt_modality][index]

        output = DatasetOutput(data=X_dict)
        # Add information about the patients, sessions etc ...
        output["participant"] = X.participant
        output["session"] = X.session
        output["age_float"] = self.df["age"][index]

        if hasattr(X, "original_image"):
            output["original_image"] = X.original_image.data.squeeze(-1)

        return output
