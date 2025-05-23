"""
Schema definition of the hierarchical config files for quick-filter model.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from aizynthmodels.utils.configs.logger import BaseLogger
from aizynthmodels.utils.configs.quick_filter.filename_postfixes import FilenamePostfixes
from aizynthmodels.utils.configs.template_based.model_hyperparams import ModelHyperparams
from aizynthmodels.utils.configs.trainer import CoreTrainer
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class CoreConfig:
    """
    The core configuration with arguments which as common across all scripts.

    The configuration for each separate script includes the parameters in the CoreConfig,
    as well as its own specific set of parameters.
    """

    logger: Optional[BaseLogger] = None
    trainer: Optional[CoreTrainer] = CoreTrainer()
    filename_postfixes: FilenamePostfixes = FilenamePostfixes()
    model_hyperparams: ModelHyperparams = ModelHyperparams()

    # Threshold used to turn probabilities into labels
    threshold: float = 0.5

    # Input
    random_seed: int = 1689
    n_predictions: int = 50
    file_prefix: Optional[str] = None  # For PrecomputedDataModule
    smiles_data_path: Optional[str] = None  # For SmilesBasedDataModule

    # Model
    model_path: Optional[str] = None
    device: str = "cuda"
    mode: str = "eval"
    clip_grad: float = 1.0
    task: str = "quick_filter"

    # Data
    datamodule: Optional[StrDict] = field(
        default_factory=lambda: {
            "type": "PrecomputedDataModule",
            "arguments": [
                {"files_prefix": "${file_prefix}"},
                {"inputs_rxn_postfix": "${filename_postfixes.model_inputs_rxn}"},
                {"inputs_prod_postfix": "${filename_postfixes.model_inputs_prod}"},
                {"labels_postfix": "${filename_postfixes.model_labels}"},
            ],
        }
    )

    # Trainer
    n_epochs: int = 50
    batch_size: int = 256
    resume: bool = False
    limit_val_batches: float = 1.0
    n_devices: int = 1
    n_nodes: int = 1
    check_val_every_n_epoch: int = 1
    acc_batches: int = 8

    # Callbacks and scorers
    callbacks: Optional[List[Any]] = None
    scorers: Optional[List[Any]] = None
