"""module for optimizing Chemformer hyperparameters"""

import logging

import hydra
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import optimize_hyperparams
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.optuna import optimize_model


@hydra.main(version_base=None, config_name="optimize_hyperparams")
@custom_config
def main(config: DictConfig) -> None:
    parameters = optimize_model(config, Chemformer)
    parameters.to_csv(config.optuna.output_hyperparams, sep="\t", index=False)
    logging.info(parameters)
    logging.info(f"Hyperparameters written to file: {config.optuna.output_hyperparams}")


if __name__ == "__main__":
    main()
