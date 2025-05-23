from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from aizynthmodels.utils.configs.logger import TensorboardLogger
from aizynthmodels.utils.configs.template_based.core import CoreConfig
from aizynthmodels.utils.configs.trainer import Trainer


@dataclass
class Train(CoreConfig):

    # Overriding from the base class
    mode: str = "train"

    logger: TensorboardLogger = TensorboardLogger()
    trainer: Trainer = Trainer()

    callbacks: List[Any] = field(
        default_factory=lambda: [
            {
                "ModelCheckpoint": [
                    {"every_n_epochs": 1},
                    {"monitor": "val_loss"},
                    {"save_last": True},
                    {"save_top_k": 3},
                ]
            },
            "ValidationScoreCallback",
        ]
    )

    scorers: List[Any] = field(
        default_factory=lambda: [{"TopKAccuracyScore": [{"top_ks": [1, 5, 10, 50]}, {"canonicalized": True}]}]
    )

    # Output
    output_directory: str = MISSING


cs = ConfigStore.instance()
cs.store(name="train", node=Train)
