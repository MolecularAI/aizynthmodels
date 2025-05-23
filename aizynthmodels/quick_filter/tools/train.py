import logging
import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.quick_filter import QuickFilter
from aizynthmodels.template_based.utils import get_filename

# flake8: noqa: F401
from aizynthmodels.utils.configs.quick_filter import train
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="train")
@custom_config
def main(config: DictConfig):
    pl.seed_everything(config.random_seed)
    logging.info("Training quick-filter model.")
    model = QuickFilter(config)
    t0 = time.time()
    model.fit()
    output_model = get_filename(config, "onnx_model")
    model.to_onnx(output_model)
    t_fit = time.time() - t0
    logging.info(f"Training complete, time: {t_fit}")
    logging.info(f"onnx model saved to: {output_model}")


if __name__ == "__main__":
    main()
