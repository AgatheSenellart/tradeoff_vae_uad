import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import nibabel as nib
import torch
from clinicadl.transforms import Transforms
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from mvae_ad.data.caps_multimodal_dataset import CapsMultimodalDataset

# create logger
logger = logging.getLogger(__name__)

# create console handler and set level to debug
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def get_train_val(
    params: Union[DictConfig, ListConfig],
) -> Tuple[CapsMultimodalDataset, CapsMultimodalDataset]:
    """
    Load the ADNI train and validation datasets.

    Args:
        params (DictConfig): Hydra configuration object containing all parameters.

    Returns:
        tuple: A tuple containing the train and validation datasets.
        Each dataset is an instance of `CapsMultimodalDataset`.

    """

    preprocessing = instantiate(params.data.preprocessing)
    extraction = instantiate(params.data.extraction.extraction)
    normalization = instantiate(params.data.normalization)
    list_txt_modalities = params.data.txt_modalities

    masks = None

    if (
        hasattr(params.data.preprocessing, "use_skull_stripped")
        and params.data.preprocessing.use_skull_stripped
    ):
        json_name = (params.data.preprocessing.reconstruction) + "_skull_stripped"
    else:
        json_name = params.data.preprocessing.reconstruction

    dict_txt_modalities_transforms = dict(params.data.txt_modalities_transforms)
    dict_txt_modalities = {
        modality: instantiate(dict_txt_modalities_transforms[modality])
        for modality in list_txt_modalities
    }

    if params.data.split is not None:
        full_split_dir = Path(
            params.paths.split_directory, "5_fold", f"split-{params.data.split}"
        )
    else:
        full_split_dir = Path(params.paths.split_directory)

    if hasattr(params.data, "use_only_baseline") and params.data.use_only_baseline:
        tsv_name = "train_baseline.tsv"
    else:
        tsv_name = "train.tsv"

    logger.info(f"TSV file used: {tsv_name}")

    train_dataset = CapsMultimodalDataset(
        caps_directory=Path(params.paths.caps_directory),
        data=Path(full_split_dir, tsv_name),
        preprocessing=preprocessing,
        transforms=Transforms(
            extraction=extraction,
            image_transforms=[normalization],
        ),
        masks=masks,
        txt_modalities=dict_txt_modalities,
        json_name=json_name,
    )

    val_dataset = CapsMultimodalDataset(
        caps_directory=Path(params.paths.caps_directory),
        preprocessing=preprocessing,
        data=Path(full_split_dir, "validation_baseline.tsv"),
        transforms=Transforms(
            extraction=extraction,
            image_transforms=[normalization],
        ),
        masks=masks,
        txt_modalities=dict_txt_modalities,
        json_name=json_name,
    )

    return (train_dataset, val_dataset)


def get_hypo_datasets(
    params: DictConfig, list_of_simulated_datasets: List[str]
) -> Dict[str, CapsMultimodalDataset]:
    """
    Load the hypo datasets specified in the configuration.

    Args:
        params (DictConfig): Hydra configuration object containing all parameters.
        list_of_simulated_datasets (List[str]): List of simulated dataset names to load.

    Returns:
        dict: A dictionary where keys are dataset names (str) and values are instances of `CapsMultimodalDataset`.
    """
    datasets_dict = {}

    for sim_group in list_of_simulated_datasets:
        data_group, pathology, perc = sim_group.split("_")

        preprocessing = instantiate(params.data.preprocessing)
        extraction = instantiate(params.data.extraction.extraction)
        normalization = instantiate(params.data.normalization)
        list_txt_modalities = params.data.txt_modalities
        hypo_sim = instantiate(
            params.data.hypo_simulation, pathology=pathology, percentage=int(perc)
        )

        if (
            hasattr(params.data.preprocessing, "use_skull_stripped")
            and params.data.preprocessing.use_skull_stripped
        ):
            json_name = (params.data.preprocessing.reconstruction) + "_skull_stripped"
        else:
            json_name = params.data.preprocessing.reconstruction

        dict_txt_modalities_transforms = dict(params.data.txt_modalities_transforms)
        dict_txt_modalities = {
            modality: instantiate(dict_txt_modalities_transforms[modality])
            for modality in list_txt_modalities
        }

        if data_group == "val":
            tsv_path = Path(
                params.paths.split_directory,
                "5_fold",
                f"split-{params.data.split}",
                "validation_baseline.tsv",
            )
        elif data_group == "test":
            tsv_path = Path(params.paths.split_directory, "test_cn_baseline.tsv")
        else:
            raise AttributeError(f"wrong dataset name :{data_group}")

        logger.info(f"TSV file used: {tsv_path}")

        datasets_dict[sim_group] = CapsMultimodalDataset(
            caps_directory=Path(params.paths.caps_directory),
            preprocessing=preprocessing,
            data=tsv_path,
            transforms=Transforms(
                extraction=extraction,
                image_transforms=[hypo_sim, normalization],
            ),
            masks=None,
            txt_modalities=dict_txt_modalities,
            json_name=json_name,
        )

    return datasets_dict


def get_test_datasets(
    params: DictConfig, list_datasets: List[str]
) -> Dict[str, CapsMultimodalDataset]:
    """
    Load the test datasets specified in the configuration.

    Args:
        params (DictConfig): Hydra configuration object containing all parameters.

    Returns:
        dict: A dictionary where keys are dataset names (str) and values are instances of `CapsMultimodalDataset`.
    """

    datasets_dict = {}

    for group in list_datasets:
        preprocessing = instantiate(params.data.preprocessing)
        extraction = instantiate(params.data.extraction.extraction)
        normalization = instantiate(params.data.normalization)
        list_txt_modalities = params.data.txt_modalities

        if (
            hasattr(params.data.preprocessing, "use_skull_stripped")
            and params.data.preprocessing.use_skull_stripped
        ):
            json_name = (params.data.preprocessing.reconstruction) + "_skull_stripped"
        else:
            json_name = params.data.preprocessing.reconstruction

        dict_txt_modalities_transforms = dict(params.data.txt_modalities_transforms)
        dict_txt_modalities = {
            modality: instantiate(dict_txt_modalities_transforms[modality])
            for modality in list_txt_modalities
        }

        tsv_path = Path(params.paths.split_directory, f"{group}.tsv")

        logger.info(f"TSV file used: {tsv_path}")

        datasets_dict[group] = CapsMultimodalDataset(
            caps_directory=Path(params.paths.caps_directory),
            preprocessing=preprocessing,
            data=tsv_path,
            transforms=Transforms(
                extraction=extraction,
                image_transforms=[normalization],
            ),
            masks=None,
            txt_modalities=dict_txt_modalities,
            json_name=json_name,
        )

    return datasets_dict


def get_brain_masks(params: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the brain mask and the hypo mask from the brain mask directory.

    Args:
        params (DictConfig): Hydra configuration object containing all parameters.

    Returns:
        torch.Tensor: A tensor containing the brain mask data.
    """

    brain_mask_path = Path(
        params.paths.mask_directory, "mni_icbm152_t1_tal_nlin_sym_09c_mask_cropped.nii"
    )
    brain_mask = torch.Tensor(nib.load(brain_mask_path).get_fdata(dtype="float32"))

    # Get the mask for computing average_precision
    mask_hypo = nib.load(
        Path(params.paths.mask_directory, "mask_hypo_ad_resampled.nii")
    )
    mask_hypo = torch.Tensor(mask_hypo.get_fdata(dtype="float32"))

    if params.data.extraction.name == "slice":
        if params.data.extraction.extraction.slice_direction == 2:
            mask_hypo = mask_hypo[:, :, params.data.extraction.extraction.slices[0]]
            brain_mask = brain_mask[:, :, params.data.extraction.extraction.slices[0]]

        else:
            raise NotImplementedError()

    return brain_mask, mask_hypo
