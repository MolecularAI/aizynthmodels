"""Module that featurizes a template library for training a template-based model"""

import logging
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
from rxnutils.chem.utils import split_rsmi
from rxnutils.data.batch_utils import read_csv_batch
from scipy import sparse

from aizynthmodels.template_based.utils import get_filename, make_and_save_input_vector
from aizynthmodels.utils.configs.template_based import pipelines  # noqa: F401
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.smiles import seq_smiles_to_fingerprint


def _make_inputs(config: DictConfig, batch: Tuple[int, int, int] = None) -> None:
    logging.info("Generating inputs...")
    smiles_dataset = read_csv_batch(
        get_filename(config, "library"),
        batch=batch,
        sep="\t",
        usecols=[config.library_config.columns.reaction_smiles],
    )

    products = np.asarray([split_rsmi(smiles)[-1] for smiles in smiles_dataset.squeeze("columns")])
    fp_kwargs = {
        "fp_length": config.model_hyperparams.fingerprint_size,
        "fp_radius": config.model_hyperparams.fingerprint_radius,
        "chirality": config.model_hyperparams.chirality,
    }
    make_and_save_input_vector(
        [products], seq_smiles_to_fingerprint, fp_kwargs, batch, get_filename(config, "model_inputs")
    )


def _make_labels(config: DictConfig, batch: Tuple[int, int, int] = None) -> None:
    logging.info("Generating labels...")

    # Find out the maximum template code
    with open(get_filename(config, "template_code"), "r") as fileobj:
        nlabels = max(int(code) for idx, code in enumerate(fileobj) if idx > 0) + 1

    template_code_data = read_csv_batch(get_filename(config, "template_code"), batch=batch)
    template_codes = template_code_data.squeeze("columns").to_numpy()

    labels = sparse.lil_matrix((len(template_codes), nlabels), dtype=np.int8)
    labels[np.arange(len(template_codes)), template_codes] = 1
    labels = labels.tocsr()

    filename = get_filename(config, "model_labels")
    if batch is not None:
        filename = filename.replace(".npz", f".{batch[0]}.npz")
    sparse.save_npz(filename, labels, compressed=True)


@hydra.main(version_base=None, config_name="expansion_model_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface for the featurization tool"""

    if config.training_pipeline.batch is not None:
        batch = tuple(config.training_pipeline.batch)
    else:
        batch = None
    _make_inputs(config, batch=batch)
    _make_labels(config, batch=batch)


if __name__ == "__main__":
    main()
