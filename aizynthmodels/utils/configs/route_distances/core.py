"""
Schema definition of the hierarchical config files for route-distances model.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

from aizynthmodels.utils.configs.logger import BaseLogger
from aizynthmodels.utils.configs.route_distances.model_hyperparams import ModelHyperparams
from aizynthmodels.utils.configs.trainer import CoreTrainer
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class CoreConfig:
    """
    The core configuration with arguments that are common across all route distances tools.

    The configuration for each separate tool includes the parameters in the CoreConfig,
    as well as its own specific set of parameters.
    """

    logger: Optional[BaseLogger] = None
    trainer: Optional[CoreTrainer] = CoreTrainer()
    trainer.precision: int = 32

    model_hyperparams: ModelHyperparams = ModelHyperparams()

    data_path: str = MISSING
    dataset_part: str = "test"

    random_seed: int = 1  # seed
    batch_size: int = 128

    # Model
    model_path: Optional[str] = None
    device: str = "cuda"
    n_devices: int = 1
    mode: str = "eval"
    task: str = "route_distances"

    # Data
    datamodule: Optional[StrDict] = field(
        default_factory=lambda: {
            "type": "TreeDataModule",
            "arguments": [{"split_part": 0.1}, {"split_seed": 1984}],
        }
    )

    # Callbacks and scorers
    callbacks: Optional[List[Any]] = None
    scorers: Optional[List[Any]] = None
