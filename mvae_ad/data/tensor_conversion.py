import argparse

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear
from clinicadl.data.datatypes.modalities.pet import ReconstructionMethod, Tracer
from clinicadl.data.datatypes.preprocessing.pet import SUVRReferenceRegion

CAPS_DIR = "SET YOUR PATH"
DATA = f"{CAPS_DIR}/splits_dsb/participants.tsv"


def convert_pet_to_tensors(
    caps_directory: str,
    reconstruction_name: str,
    participants_tsv: str,
    use_skull_stripped: bool,
):
    """
    Convert PET data to tensors.

    Args:
        caps_directory (str): Path to the CAPS directory.
        reconstruction_name (str): Name of the PET reconstruction.
        participants_tsv (str): Path to the TSV file containing participant data.
        use_skull_stripped (bool): Whether to use skull-stripped data.

    Raises:
        ValueError: If the reconstruction name is not recognized.

    """

    if reconstruction_name not in ["coregiso", "coregavg"]:
        raise ValueError(
            f"Reconstruction name '{reconstruction_name}' is not recognized. "
            "Please use 'coregiso' or 'coregavg'."
        )

    preprocessing = PETLinear(
        suvr_reference_region=SUVRReferenceRegion.CEREBELLUM_PONS2,
        reconstruction=ReconstructionMethod(reconstruction_name),
        tracer=Tracer.FDG,
        use_skull_stripped=use_skull_stripped,
    )

    dataset = CapsDataset(
        caps_directory=caps_directory,
        preprocessing=preprocessing,
        data=participants_tsv,
        masks=["desc-brain_dseg"] if use_skull_stripped else None,
    )

    conversion_name = (reconstruction_name) + (
        "_skull_stripped" if use_skull_stripped else ""
    )

    dataset.to_tensors(conversion_name=conversion_name, save_transforms=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PET data to tensors.")
    parser.add_argument(
        "--reconstruction_name",
        type=str,
        choices=["coregiso", "coregavg"],
        default='coregiso',
        help="Name of the PET reconstruction (coregiso or coregavg).",
    )
    parser.add_argument(
        "--caps_directory",
        type=str,
        default=CAPS_DIR,
        help="Path to the CAPS directory containing PET data.",
    )
    parser.add_argument(
        "--participants_tsv",
        type=str,
        default=DATA,
        help="Path to the TSV file containing participant data.",
    )
    parser.add_argument(
        "--use_skull_stripped",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    caps_directory = args.caps_directory
    participants_tsv = args.participants_tsv
    reconstruction_name = args.reconstruction_name
    use_skull_stripped = args.use_skull_stripped

    convert_pet_to_tensors(
        caps_directory, reconstruction_name, participants_tsv, use_skull_stripped
    )
