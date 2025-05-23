"""
This module contains scripts to generate metadata artifacts for the filter policy

* A concatenated library file with both positive and negative data
* A NPZ file with indices of the training, validation and testing partitions

"""

import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.configs.quick_filter import pipelines  # noqa: F401
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.pipelines.data_utils import split_data


def _save_split_indices(dataset: pd.DataFrame, config: DictConfig) -> None:
    """Perform a split and save the indices to disc"""
    logging.info("Creating split of dataset...")

    test_indices = []
    train_indices = []
    val_indices = []
    # Splitting positive and negative data separately
    for label in [1, 0]:
        split_data(
            dataset.query(f"{config.library_columns.label}=={label}"),
            train_frac=config.training_fraction,
            random_seed=config.random_seed,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    logging.info(f"Selecting {len(train_indices)} ({len(train_indices)/len(dataset)*100:.2f}%) records as training set")
    logging.info(f"Selecting {len(val_indices)} ({len(val_indices)/len(dataset)*100:.2f}%) records as validation set")
    logging.info(
        f"Selecting {len(test_indices)} ({len(test_indices)/len(dataset)*100:.2f}%) records as test set",
    )

    np.savez(
        get_filename(config, "split_indices"),
        train=train_indices,
        val=val_indices,
        test=test_indices,
    )


@hydra.main(version_base=None, config_name="filter_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface of the routines"""
    pos_dataset = pd.read_csv(
        get_filename(config, "template_library"),
        sep="\t",
    )
    pos_dataset = pos_dataset.assign(**{config.library_columns.label: 1})

    neg_dataset = pd.read_csv(
        get_filename(config, "generated_library"),
        sep="\t",
    )
    neg_dataset = neg_dataset.assign(**{config.library_columns.label: 0})

    # Combine positive and negative data
    renaming_map = {
        config.template_library_columns.reaction_smiles: config.library_columns.reaction_smiles,
        config.template_library_columns.reaction_hash: config.library_columns.reaction_hash,
    }
    dataset = pd.concat([pos_dataset, neg_dataset]).rename(columns=renaming_map)
    dataset = dataset[
        [config.library_columns.reaction_smiles, config.library_columns.reaction_hash, config.library_columns.label]
    ]
    # To "mix" positive and negative data
    dataset = dataset.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
    dataset.to_csv(get_filename(config, "library"), sep="\t", index=False)

    _save_split_indices(dataset, config)


if __name__ == "__main__":
    main()
