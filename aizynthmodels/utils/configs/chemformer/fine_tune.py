from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.chemformer.core import CoreConfig
from aizynthmodels.utils.configs.chemformer.model_hyperparams import TrainModelHyperparams
from aizynthmodels.utils.configs.logger import TensorboardLogger
from aizynthmodels.utils.configs.trainer import Trainer
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class FineTune(CoreConfig):

    # Overriding from the base class
    logger: TensorboardLogger = TensorboardLogger()
    trainer: Trainer = Trainer()

    model_hyperparams: TrainModelHyperparams = TrainModelHyperparams()

    batch_size: int = 64
    n_predictions: int = 1

    task: str = "backward_prediction"  # Name of task.
    # Note: for the SynthesisDatamodule, 'task'
    # has to be one of ["forward_prediction", "backward_prediction", "mol_opt"]
    mode: str = "train"

    callbacks: List[Any] = field(
        default_factory=lambda: [
            "LearningRateMonitor",
            {
                "ModelCheckpoint": [
                    {"every_n_epochs": 1},
                    {"monitor": "validation_loss"},
                    {"save_last": True},
                    {"save_top_k": 3},
                ]
            },
            "ValidationScoreCallback",
        ]
    )

    masker: Optional[StrDict] = field(default_factory=lambda: {"type": None, "arguments": []})

    scorers: Optional[List[Any]] = field(default_factory=lambda: ["FractionInvalidScore", "TopKAccuracyScore"])

    # Class-specific arguments
    seed: int = 73
    accelerator: Optional[str] = None

    # Output
    output_directory: str = "chemformer"


cs = ConfigStore.instance()
cs.store(name="fine_tune", node=FineTune)
