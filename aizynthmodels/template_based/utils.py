"""Module containing utility routines for various manipulations"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig
from scipy import sparse


def get_filename(config: DictConfig, name: str, subset: str = "") -> str:
    """
    From a config file and a name extract the full filename

    :param config: the configuration
    :param name: the data item to extract filename
    :param subset: an optional subset of the item
    :returns: the full filename
    """
    name_value = config.filename_postfixes.get(name, name)
    if subset:
        name_value = subset + "_" + name_value
    return config.file_prefix + "_" + name_value


def make_and_save_input_vector(
    input_list: Sequence[Sequence[str]],
    featurizer: Callable,
    fp_kwargs: Dict[str, Any],
    batch: Optional[Tuple[int, int, int]],
    filename: str,
):
    """
    Apply a featurizer across the axsis of a numpy array in order to create fingerprints
    for either molecules or reactions and then save it as a compressed matrix to file

    :param input_lists: the list of input SMILES, the dimension should match the featurizer
    :param featurizer: a callable that takes a sequence of SMILES and the keyword arguments
                       and make a fingerprint vector. These are in `aizynthmodels.utils.smiles`
    :param fp_kwargs: the keyword argument for the featurizer
    :param batch: a batch specification, controlling the filename
    :param filename: the base filename of the output file
    """
    inputs = np.apply_along_axis(featurizer, 0, input_list, **fp_kwargs)
    inputs = sparse.lil_matrix(inputs.T).tocsr()
    if batch is not None:
        filename = filename.replace(".npz", f".{batch[0]}.npz")
    sparse.save_npz(filename, inputs, compressed=True)
