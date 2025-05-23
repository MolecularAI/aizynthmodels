import logging
import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.template_based import TemplateBasedRetrosynthesis

# flake8: noqa: F401
from aizynthmodels.utils.configs.template_based import inference_score
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="inference_score")
@custom_config
def main(config: DictConfig):
    pl.seed_everything(config.random_seed)
    logging.info("Predicting with template-based retrosynthesis model.")
    model = TemplateBasedRetrosynthesis(config)
    t0 = time.time()
    model.score_model()
    t_fit = time.time() - t0
    logging.info(f"Inference complete, time: {t_fit}")


if __name__ == "__main__":
    main()
