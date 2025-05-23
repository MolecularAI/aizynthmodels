""" Module for making predictions of (compressed) route distance matrix """

import logging

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.route_distances import RouteDistanceModel

# flake8: noqa: F401
from aizynthmodels.utils.configs.route_distances import predict
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="predict")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.random_seed)

    model = RouteDistanceModel(config)
    predictions = model.predict()

    if not config.output_predictions:
        return predictions
    predictions_df = pd.DataFrame({"distances_compressed": np.asarray(predictions["predictions"])})
    predictions_df.to_csv(config.output_predictions, sep="\t", index=False)
    logging.info("Prediction done!")


if __name__ == "__main__":
    main()
