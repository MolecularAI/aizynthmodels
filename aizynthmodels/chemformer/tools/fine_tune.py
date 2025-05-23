import logging
import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import fine_tune
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="fine_tune")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)
    logging.info("Fine-tuning CHEMFORMER.")
    chemformer = Chemformer(config)
    t0 = time.time()
    chemformer.fit()
    t_fit = time.time() - t0
    logging.info(f"Training complete, time: {t_fit}")
    logging.info("Done fine-tuning.")
    return


if __name__ == "__main__":
    main()
