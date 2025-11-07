from multivae.trainers.base.callbacks import MLFlowCallback, TensorboardCallback
import hydra, os


def setup_logger(params, trainer_config, model_config, PROJECT_NAME="CVAE_ADNI"):
    if params.logger == "mlflow":
        logger = MLFlowCallback()
        logger.setup(
            trainer_config,
            model_config,
            project_name=PROJECT_NAME,
            logging_dir=params.paths.logging_directory,
        )

        # Log any remaining parameters
        for k, item in params.items():
            if k not in trainer_config.to_dict() and k not in model_config.to_dict():
                logger._mlflow.log_param(k, item)

    elif params.logger == "tensorboard":
        logger = TensorboardCallback()
        logger.setup(
            logging_dir=os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                "tensorboard_logs",
            )
        )

    else:
        raise AttributeError(
            f"Logger {params.logger} not implemented. Please choose between 'mlflow' and 'tensorboard'."
        )
    return logger
