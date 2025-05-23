import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import predict
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.writing import predictions_to_file


@hydra.main(version_base=None, config_name="predict")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)
    chemformer = Chemformer(config)

    logging.info("Making predictions...")
    predictions = chemformer.predict()
    predictions_to_file(
        config.output_predictions,
        predictions["predictions"],
        predictions["log_likelihoods"],
        predictions.get("ground_truth"),
        prediction_col="sampled_smiles",
        ranking_metric_col="log_likelihood",
    )
    logging.info("Finished predictions.")
    return


if __name__ == "__main__":
    main()
