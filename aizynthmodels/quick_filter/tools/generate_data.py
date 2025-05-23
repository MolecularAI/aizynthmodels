""" Tool to generate negative data from a template library for training the quick-filter model
"""

import logging
from collections import defaultdict
from functools import partial

import hydra
import pandas as pd
from omegaconf import DictConfig
from rxnutils.data.batch_utils import read_csv_batch
from tqdm import tqdm

from aizynthmodels.quick_filter.generators import random_template_selection, strict_template_selection
from aizynthmodels.template_based.utils import get_filename

# flake8: noqa: F401
from aizynthmodels.utils.configs.quick_filter import pipelines
from aizynthmodels.utils.hydra import custom_config

tqdm.pandas()


@hydra.main(version_base=None, config_name="filter_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface to the routines"""
    if config.batch is not None:
        batch = tuple(config.batch)
    else:
        batch = None

    full_dataset = pd.read_csv(
        get_filename(config, "template_library"),
        sep="\t",
        usecols=[config.template_library_columns.template_hash, config.template_library_columns.retro_template],
    )
    batch_dataset = read_csv_batch(
        get_filename(config, "template_library"),
        batch=batch,
        sep="\t",
    )

    TYPE2METHOD = {
        "random": partial(
            random_template_selection,
            nsamples=config.negative_generation.random_samples,
            random_state=config.negative_generation.random_state,
            full_df=full_dataset,
        ),
        "strict": strict_template_selection,
    }

    name_config = config.template_library_columns
    neg_data = defaultdict(list)
    for type_ in set(config.negative_generation.type):
        n_added = batch_dataset.apply(TYPE2METHOD[type_], name_config=name_config, negative_data=neg_data, axis=1)
        logging.info(f"Added {n_added.sum()} for method {type_}")

    neg_data = pd.DataFrame(neg_data).rename(columns=dict())
    logging.info(f"Generate in total {len(neg_data)} rows of negative data")
    if len(neg_data) > 0:
        neg_data = neg_data.drop_duplicates("reaction_hash")
        sel = neg_data[name_config.reaction_hash].isin(batch_dataset[name_config.reaction_hash])
        neg_data = neg_data[~sel]
    else:
        neg_data = pd.DataFrame({col: [] for col in batch_dataset.columns})
    logging.info(f"Keeping {len(neg_data)} data points not in the template library")

    filename = get_filename(config, "generated_library")
    if batch is not None:
        filename = filename.replace(".csv", f".csv.{batch[0]}")
    neg_data.to_csv(filename, index=False, sep="\t")


if __name__ == "__main__":
    main()
