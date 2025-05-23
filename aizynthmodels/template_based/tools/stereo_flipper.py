"""Module containing routines to flip stereo centers for specific reactions"""

import re
from functools import partial

import hydra
import pandas as pd
from omegaconf import DictConfig
from rxnutils.chem.template import ReactionTemplate
from rxnutils.data.batch_utils import read_csv_batch

# flake8: noqa: F401
from aizynthmodels.utils.configs.template_based import stereo_flipper
from aizynthmodels.utils.hydra import custom_config

# One or two @ characters in isolation
STEREOCENTER_REGEX = r"[^@]([@]{1,2})[^@]"


def _count_chiral_centres(row: pd.Series) -> int:
    prod_template = row.split(">>")[0]
    return len(re.findall(STEREOCENTER_REGEX, prod_template))


def _flip_chirality(row: pd.Series, template_column: str, template_hash_column: str) -> pd.Series:
    """
    Change @@ to @ and vice versa in a retrosynthesis template
    and then create a new template and a hash for that template
    """
    dict_ = row.to_dict()
    prod_template = row[template_column].split(">>")[0]
    nats = len(re.search(STEREOCENTER_REGEX, prod_template)[1])
    assert nats in [1, 2]
    if nats == 1:
        dict_[template_column] = row[template_column].replace("@", "@@")
    else:
        dict_[template_column] = row[template_column].replace("@@", "@")
    dict_[template_hash_column] = ReactionTemplate(dict_[template_column], direction="retro").hash_from_bits()
    return pd.Series(dict_)


def flip_stereo(
    data: pd.DataFrame, selection_query: str, template_column: str, template_hash_column: str
) -> pd.DataFrame:
    """
    Find templates with one stereo center and flip it thereby creating
    new templates. These templates are appended onto the existing dataframe.

    A column "FlippedStereo" will be added to indicate if a stereocenter
    was flipped.

    :param data: the template library
    :param selection_query: only flip the stereo for a subset of rows
    :param template_column: reaction template column name
    :param template_hash_column: template hash column name
    :returns: the concatenated dataframe.
    """
    sel_data = data.query(selection_query)
    sel_data = sel_data[~sel_data[template_column].isna()]

    chiral_centers_count = sel_data[template_column].apply(_count_chiral_centres)
    sel_flipping = chiral_centers_count == 1

    flip_fcn = partial(_flip_chirality, template_column=template_column, template_hash_column=template_hash_column)
    flipped_data = sel_data[sel_flipping].apply(flip_fcn, axis=1)

    existing_hashes = set(sel_data[template_hash_column])
    keep_flipped = flipped_data[template_hash_column].apply(lambda hash_: hash_ not in existing_hashes)
    flipped_data = flipped_data[keep_flipped]

    all_data = pd.concat([data, flipped_data])
    flag_column = [False] * len(data) + [True] * len(flipped_data)
    return all_data.assign(FlippedStereo=flag_column)


@hydra.main(version_base=None, config_name="stereo_flipper")
@custom_config
def main(config: DictConfig) -> None:
    data = read_csv_batch(config.input_data, sep="\t", index_col=False, batch=config.batch)
    data = flip_stereo(data, config.query, config.template_column, config.template_hash_column)
    data.to_csv(config.output_data, index=False, sep="\t")


if __name__ == "__main__":
    main()
