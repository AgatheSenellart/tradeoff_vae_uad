import torch
from multivae.models import AutoModel

from mvae_ad.metrics.metrics import reconstruction_error_depending_on_pixel_values

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--hydra_path", type=str, required=True)
    hydra_path = parser.parse_args().hydra_path

    # load the hydra config
    from mvae_ad.metrics.metrics import get_hydra_config_and_model

    params, best_model_path = get_hydra_config_and_model(hydra_path)

    # Get the datasets
    from mvae_ad.data.dataset_handler import get_brain_masks, get_train_val

    _, val_dataset = get_train_val(params)

    # Get the masks
    brain_mask, hypo_mask = get_brain_masks(params)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    # Evaluate the model
    reconstruction_error_depending_on_pixel_values(
        eval_dataset=val_dataset,
        model=AutoModel.load_from_folder(best_model_path),
        output_dir=best_model_path,
        device=device,
        n_images=min(30, len(val_dataset)),
    )
