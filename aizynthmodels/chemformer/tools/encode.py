"""Module for obtaining the latent space of Chemformer (encoder layer output)."""

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import encode
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="predict")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)
    chemformer = Chemformer(config)

    logging.info("Encoding SMILES")
    encodings = chemformer.encode(dataset=config.dataset_part)

    return encodings


if __name__ == "__main__":
    main()
