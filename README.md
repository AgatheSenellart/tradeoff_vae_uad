# Mitigating the Reconstruction-Detection Trade-off in VAE-based Unsupervised Anomaly Detection

This is the GitHub repository for the paper entitled "Mitigating the Reconstruction-Detection Trade-off in VAE-based Unsupervised Anomaly Detection", currently in submission.

## Installation

1. Create a new environment, for instance:

   ```shell
   conda create -n uad_env python=3.11
   ```

2. Clone this repository, and install its dependencies:

    ```shell
    git clone [name] && cd [name]
    pip install -e .
    ```

3. Clone the MultiVae repository and install it in developer mode from the branch `adni_dev`:

   ```shell
   git clone https://github.com/AgatheSenellart/MultiVae.git
   ```
   From inside the MultiVae folder:
   ```
   git switch adni_dev
   pip install -e .
   ```

## Data

Our experiment use data from the [ADNI database](https://adni.loni.usc.edu/). 
The dataset is preprocessed with Clinica. The output of the preprocessing is a [CAPS directory](https://aramislab.paris.inria.fr/clinica/docs/public/dev/CAPS/Introduction/). 

To extract tensors from this dataset, you can use the mvae_ad/data/tensor_conversion.py script.

## Training $\beta$-VAE models

Models can be trained using:

```py
python mvae_ad/train_vae.py model.config.beta=10 architecture.latent_dim=64 seed=0 
```

The configuration files can be changed in the folder `configs/configs_training` or by directly overriding a parameter in the command line (ex: 'seed=1').

⚠️ Please make sure to adjust the paths to the data in `paths`.

After training all the results as well as the best model is saved in 'outputs'. You will find there:

- reconstruction and anomaly detection metrics for all datasets (val, test_AD_30, test_AD_50, test_ad_baseline, test_cn_baseline)

- the latent distances between distributions in "wasserstein_dist.csv"

### Using beta-scheduling

Models can be trained using:

```py
python mvae_ad/train_vae.py trainer=beta_scheduling model.config.beta=30 data.split=0
```

### Sparse-VAE

Models can be trained using 

```
python mvae_ad/train_vae.py model=sparse data.split=0
```

## Monitoring
Tensorboard is used for monitoring. 
You can see the training curve by running 
```
pip install tensorboard
tensorboard --logdir=outputs
```

## Analysis

The code (jupyter notebooks) used to analyse the results and obtain the Figures and Table in the paper can be found in the folder `mvae_ad/plots`.
