"""Module routines for splitting data for quick-filter model"""

import logging
from typing import Union

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import sparse

from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.configs.quick_filter import pipelines  # noqa: F401
from aizynthmodels.utils.hydra import custom_config


def _split_and_save_data(
    data: Union[pd.DataFrame, np.ndarray, sparse.csr_matrix],
    data_label: str,
    config: DictConfig,
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    test_arr: np.ndarray,
) -> None:
    array_dict = {"training": train_arr, "validation": val_arr, "testing": test_arr}
    for subset, arr in array_dict.items():
        filename = get_filename(config, data_label, subset=subset)
        if isinstance(data, pd.DataFrame):
            data.iloc[arr].to_csv(
                filename,
                sep="\t",
                index=False,
            )
        elif isinstance(data, np.ndarray):
            np.savez(filename, data[arr])
        else:
            sparse.save_npz(filename, data[arr], compressed=True)


@hydra.main(version_base=None, config_name="filter_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface for the splitting routines"""

    split_indices = np.load(get_filename(config, "split_indices"))
    train_arr = split_indices["train"]
    val_arr = split_indices["val"]
    test_arr = split_indices["test"]

    logging.info("Splitting filter library...")
    dataset = pd.read_csv(
        get_filename(config, "library"),
        sep="\t",
    )
    _split_and_save_data(dataset, "library", config, train_arr, val_arr, test_arr)

    logging.info("Splitting labels...")
    data = np.load(get_filename(config, "model_labels"))["arr_0"]
    _split_and_save_data(data, "model_labels", config, train_arr, val_arr, test_arr)

    logging.info("Splitting inputs...")
    data = sparse.load_npz(get_filename(config, "model_inputs_prod"))
    _split_and_save_data(data, "model_inputs_prod", config, train_arr, val_arr, test_arr)
    data = sparse.load_npz(get_filename(config, "model_inputs_rxn"))
    _split_and_save_data(data, "model_inputs_rxn", config, train_arr, val_arr, test_arr)


if __name__ == "__main__":
    main()
