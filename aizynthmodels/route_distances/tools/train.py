""" Module for CLI tool to train LSTM-based model """

import logging
import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from aizynthmodels.route_distances import RouteDistanceModel

# flake8: noqa: F401
from aizynthmodels.utils.configs.route_distances import train
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="train")
@custom_config
def main(config: DictConfig) -> None:
    logging.info("Training route-distance LSTM model.")
    logging.info(OmegaConf.to_yaml(config))

    pl.seed_everything(config.random_seed)

    model = RouteDistanceModel(config)
    t0 = time.time()
    model.fit()
    t_fit = time.time() - t0
    logging.info(f"Training complete, time: {t_fit}")


if __name__ == "__main__":
    main()
