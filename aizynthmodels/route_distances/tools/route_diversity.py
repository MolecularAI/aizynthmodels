""" Module for computing route diversity for targets """

import logging

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from aizynthmodels.route_distances import RouteDistanceModel

# flake8: noqa: F401
from aizynthmodels.utils.configs.route_distances import predict
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.type_utils import StrDict


def self_diversity(model: RouteDistanceModel, datamodule_config: StrDict, finder_output: pd.DataFrame):
    """
    Calculate route diversity of solved routes for each target molecule
    by mean distance between the routes.

    :param model: RouteDistance model used to predict distances.
    :param datamodule_config: Config for datamodule. Will be updated with route lists for each target.
    :param finder_output: Dataframe with aizynthfinder outputs, incl. routes (trees).
    """
    mean_distances = np.nan * np.zeros(finder_output.shape[0])

    counter = 0
    for _, row in finder_output.iterrows():

        trees_solved = [tree for tree in row.trees if tree["metadata"]["is_solved"]]

        if len(trees_solved) == 0:
            counter += 1
            continue

        datamodule_config["arguments"] = [{"route_list": trees_solved}]
        model.set_datamodule(datamodule_config=datamodule_config)

        predictions = model.predict()
        mean_distances[counter] = np.mean(predictions["predictions"]) if predictions["predictions"] else 0
        counter += 1

    return mean_distances


@hydra.main(version_base=None, config_name="predict")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.random_seed)

    finder_output = pd.read_json(config.data_path, orient="table")
    config.data_path = ""

    datamodule_config = OmegaConf.to_container(config.datamodule, resolve=True)
    config.datamodule = None

    model = RouteDistanceModel(config)
    diversity = self_diversity(model, datamodule_config, finder_output)

    if not config.output_predictions:
        return diversity

    predictions_df = pd.DataFrame({"self_diversity": diversity})
    predictions_df.to_csv(config.output_predictions, sep="\t", index=False)
    logging.info("Diversity prediction done!")


if __name__ == "__main__":
    main()
