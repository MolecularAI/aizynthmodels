"""
This module contains scripts to generate metadata artifacts for a template library

* A one-column CSV file with generated template codes, used for featurization
* A gzipped CSV file with unique templates and metadata to be used with aizynthfinder
* A JSON file with extensive metadata and lookup to reference ID underlying the templates
* A NPZ file with indices of the training, validation and testing partitions

"""

import json
import logging
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.configs.template_based import pipelines  # noqa: F401
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.pipelines.data_utils import extract_route_reactions, split_data


def _add_and_save_template_code(dataset: pd.DataFrame, config: DictConfig) -> None:
    columns = config.library_config.columns
    template_labels = LabelEncoder()
    dataset.loc[:, columns.template_code] = template_labels.fit_transform(dataset[columns.template_hash])

    logging.info("Saving template codes...")
    dataset[columns.template_code].to_csv(get_filename(config, "template_code"), index=False)


def _make_template_dataset(dataset: pd.DataFrame, config: DictConfig) -> None:
    def group_apply(
        group: pd.DataFrame,
        id_lookup: Dict[str, List[str]],
        columns: DictConfig,
        template_set: str,
        metadata_columns: List[str],
    ) -> pd.Series:
        if id_lookup:
            refs = sorted(id_ for pseudo_hash in group[columns.reaction_hash] for id_ in id_lookup[pseudo_hash])
        else:
            refs = []
        dict_ = {
            "_id": group[columns.template_hash].iloc[0],
            "count": len(refs),
            "dimer_only": False,
            "index": group[columns.template_code].iloc[0],
            "intra_only": False,
            "necessary_reagent": "",
            "reaction_smarts": group[columns.retro_template].iloc[0],
            "references": refs,
            "template_set": template_set,
        }
        for column in metadata_columns:
            dict_[column] = group[column].iloc[0]
        return pd.Series(dict_)

    selected_ids_path = config.training_pipeline.get("selected_ids_path")
    if selected_ids_path:
        with open(selected_ids_path, "r") as fileobj:
            pseudo_hash_to_id = json.load(fileobj)
    else:
        pseudo_hash_to_id = None

    columns = config.library_config.columns
    lookup = dataset.groupby(columns.template_hash).apply(
        group_apply,
        id_lookup=pseudo_hash_to_id,
        columns=columns,
        template_set=config.library_config.template_set,
        metadata_columns=config.library_config.metadata_columns,
    )
    return lookup


def _save_lookup(template_dataset: pd.DataFrame, config: DictConfig) -> None:
    """
    Save a JSON file with template data and reference to original reaction.
    This was intended for an internal platform originally, hence the formatting.
    """
    logging.info("Creating template hash lookup...")
    template_dataset2 = template_dataset.drop(columns=config.library_config.metadata_columns)
    with open(get_filename(config, "template_lookup"), "w") as fileobj:
        json.dump(
            template_dataset2.to_dict(orient="records"),
            fileobj,
        )


def _save_unique_templates(template_dataset: pd.DataFrame, config: DictConfig) -> None:
    """
    Save a gzipped CSV file with template code, template SMARTS and metadata
    that can be used by AiZynthFinder
    """
    logging.info("Creating unique template library...")
    columns = config.library_config.columns
    template_dataset2 = template_dataset[
        ["index", "reaction_smarts"] + config.library_config.metadata_columns + ["count"]
    ]
    template_dataset2 = template_dataset2.rename(
        columns={
            "index": columns.template_code,
            "reaction_smarts": columns.retro_template,
            "count": columns.library_occurrence,
        },
    )
    template_dataset2.to_csv(get_filename(config, "unique_templates"), index=False, sep="\t")


def _save_split_indices(dataset: pd.DataFrame, config: DictConfig) -> None:
    """Perform a stratified split and save the indices to disc"""
    logging.info("Creating split of template library...")
    columns = config.library_config.columns

    if config.training_pipeline.routes_to_exclude:
        reaction_hashes = extract_route_reactions(config.training_pipeline.routes_to_exclude)
        logging.info(
            f"Found {len(reaction_hashes)} unique reactions given routes. Will make these test set",
        )
        sel = dataset[columns.reaction_hash].isin(reaction_hashes)
        test_indices = dataset[sel].index.to_list()
        subdata = dataset[~sel]
    else:
        test_indices = []
        subdata = dataset
    train_indices = []
    val_indices = []

    subdata.groupby(columns.template_hash).apply(
        split_data,
        train_frac=config.training_pipeline.training_fraction,
        random_seed=config.training_pipeline.random_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=None if test_indices else test_indices,
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


@hydra.main(version_base=None, config_name="expansion_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface ot the routines"""

    dataset = pd.read_csv(
        get_filename(config, "library"),
        sep="\t",
    )

    _add_and_save_template_code(dataset, config)
    template_dataset = _make_template_dataset(dataset, config)

    _save_lookup(template_dataset, config)
    _save_unique_templates(template_dataset, config)

    _save_split_indices(dataset, config)


if __name__ == "__main__":
    main()
