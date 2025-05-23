"""Module for splitting a Chemformer dataset into train / validation / test sets"""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.pipelines.data_utils import extract_route_reactions, split_data


def split_dataset(config: DictConfig, dataset: pd.DataFrame) -> None:
    if config.preprocessing_pipeline.routes_to_exclude:
        reaction_hashes = extract_route_reactions(config.preprocessing_pipeline.get("routes_to_exclude", []))
        logging.info(
            f"Found {len(reaction_hashes)} unique reactions given routes. Will make these test set",
        )
        is_external = dataset[config.preprocessing_pipeline.reaction_hash_col].isin(reaction_hashes)
        subdata = dataset[~is_external]
    else:
        is_external = None
        subdata = dataset
    train_indices = []
    val_indices = []
    test_indices = []

    subdata.apply(
        split_data,
        train_frac=config.preprocessing_pipeline.training_fraction,
        random_seed=config.preprocessing_pipeline.seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    subdata[config.preprocessing_pipeline.set_col] = "train"
    subdata.loc[val_indices, config.preprocessing_pipeline.set_col] = "val"
    subdata.loc[test_indices, config.preprocessing_pipeline.set_col] = "test"

    if is_external is None:
        subdata.to_csv(config.chemformer_data_path, sep="\t", index=False)
        return

    dataset.loc[~is_external, config.preprocessing_pipeline.set_col] = subdata.set.values
    dataset.loc[is_external, config.preprocessing_pipeline.set_col] = "test"

    dataset[config.preprocessing_pipeline.is_external_col] = False
    dataset.loc[is_external, config.preprocessing_pipeline.is_external_col] = True
    dataset.to_csv(config.chemformer_data_path, sep="\t", index=False)


@hydra.main(version_base=None, config_path="../../../utils/chemformer/pipelines", config_name="data_prep")
@custom_config
def main(config: DictConfig):
    """Command-line interface to the routines"""
    dataset = pd.read_csv(config.preprocessing_pipeline.reaction_components_path, sep="\t")
    split_dataset(config, dataset)
    return


if __name__ == "__main__":
    main()
