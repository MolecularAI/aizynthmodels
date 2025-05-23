import logging
import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.template_based import TemplateBasedRetrosynthesis

# flake8: noqa: F401
from aizynthmodels.utils.configs.template_based import predict
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.writing import predictions_to_file


@hydra.main(version_base=None, config_name="predict")
@custom_config
def main(config: DictConfig):
    pl.seed_everything(config.random_seed)
    logging.info("Predicting with template-based retrosynthesis model.")
    model = TemplateBasedRetrosynthesis(config)
    t0 = time.time()
    predictions = model.predict()
    t_fit = time.time() - t0
    logging.info(f"Inference complete, time: {t_fit}")
    predictions_to_file(
        config.output_predictions,
        predictions["predictions"],
        predictions["probabilities"],
        predictions["ground_truth"],
    )


if __name__ == "__main__":
    main()
