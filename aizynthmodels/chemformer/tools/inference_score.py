import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import inference_score
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="inference_score")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)

    logging.info("Running model inference and scoring.")

    chemformer = Chemformer(config)
    chemformer.score_model()
    logging.info("Model inference and scoring done.")
    return


if __name__ == "__main__":
    main()
