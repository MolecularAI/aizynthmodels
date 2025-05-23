"""
Schema definition and validation of the hierarchical config files.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

from aizynthmodels.utils.configs.chemformer.model_hyperparams import InferenceModelHyperparams
from aizynthmodels.utils.configs.logger import BaseLogger
from aizynthmodels.utils.configs.trainer import CoreTrainer
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class CoreConfig:
    """
    The core configuration with arguments which as common across all Chemformer scripts.

    The configuration for each separate script includes the parameters in the CoreConfig,
    as well as its own specific set of parameters.
    """

    logger: Optional[BaseLogger] = None
    trainer: Optional[CoreTrainer] = CoreTrainer()

    model_hyperparams: InferenceModelHyperparams = InferenceModelHyperparams()

    # Input
    data_path: str = MISSING
    vocabulary_path: str = MISSING
    seed: int = 1

    # Model
    model_path: Optional[str] = None
    model: StrDict = field(
        default_factory=lambda: {
            "type": "BARTModel",
            "arguments": [{"ckpt_path": "${model_path}"}],
        }
    )
    device: str = "cuda"
    mode: str = "eval"
    clip_grad: float = 1.0
    task: str = "forward_prediction"

    # Data
    augmentation_probability: float = 0.0
    augmentation_strategy: Optional[str] = (
        None  # Can be set to "all", "reactants", "products" when using synthesis datamodule
    )
    datamodule: Optional[StrDict] = field(
        default_factory=lambda: {
            "type": "SynthesisDataModule",
            "arguments": [{"augmentation_strategy": "${augmentation_strategy}"}],
        }
    )

    masker: Optional[StrDict] = None

    # Trainer
    n_epochs: int = 50
    batch_size: int = 128
    resume: bool = False
    limit_val_batches: float = 1.0
    n_devices: int = 1
    n_nodes: int = 1
    check_val_every_n_epoch: int = 1
    acc_batches: int = 8

    # Sampler
    sampler: Optional[str] = "BeamSearchSampler"
    sample_unique: bool = False
    n_predictions: int = 10

    # Callbacks and scorers
    callbacks: Optional[List[Any]] = None
    scorers: Optional[List[Any]] = None
