from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.logger import CSVLogger
from aizynthmodels.utils.configs.route_distances.core import CoreConfig
from aizynthmodels.utils.configs.trainer import Trainer


@dataclass
class Train(CoreConfig):
    logger: CSVLogger = CSVLogger()
    trainer: Trainer = Trainer()
    trainer.precision: int = 32

    mode: str = "train"
    n_epochs: int = 50
    limit_val_batches: float = 1.0
    n_devices: int = 1
    n_nodes: int = 1
    check_val_every_n_epoch: int = 1
    acc_batches: int = 4
    clip_grad: float = 1.0

    output_directory: str = "csv_logs/"

    callbacks: List[Any] = field(
        default_factory=lambda: [
            "LearningRateMonitor",
            {
                "ModelCheckpoint": [
                    {"every_n_epochs": 1},
                    {"monitor": "val_loss"},
                    {"save_last": True},
                    {"save_top_k": 3},
                ]
            },
        ]
    )
    scorers: List[Any] = field(default_factory=lambda: ["MeanAbsoluteError"])


cs = ConfigStore.instance()
cs.store(name="train", node=Train)
