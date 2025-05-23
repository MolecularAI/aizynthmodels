from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.chemformer.core import CoreConfig
from aizynthmodels.utils.configs.chemformer.model_hyperparams import TrainModelHyperparams
from aizynthmodels.utils.configs.logger import TensorboardLogger
from aizynthmodels.utils.configs.trainer import Trainer
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class Pretrain(CoreConfig):

    # Overriding from the base class
    logger: TensorboardLogger = TensorboardLogger()
    trainer: Trainer = Trainer()

    model_hyperparams: TrainModelHyperparams = TrainModelHyperparams()
    mode: str = "train"
    acc_batches: int = 1
    n_epochs: int = 10
    task: str = "mask_aug"
    n_predictions: int = 1
    batch_size: int = 64

    callbacks: List[Any] = field(
        default_factory=lambda: [
            "LearningRateMonitor",
            {
                "ModelCheckpoint": [
                    {"every_n_epochs": 1},
                    {"monitor": "validation_loss"},
                ]
            },
            "ValidationScoreCallback",
            {"StepCheckpoint": [{"step_interval": 50000}]},
        ]
    )

    # Available datamodules for pre-training:
    #   - aizynthmodels.chemformer.data.mol_data.ChemblDataModule
    #   - aizynthmodels.chemformer.data.mol_data.ZincDataModule
    datamodule: Optional[StrDict] = field(
        default_factory=lambda: {
            "type": "aizynthmodels.chemformer.data.ZincDataModule",
            "arguments": [],
        }
    )

    # Args specific to pre-training
    seed: int = 37

    masker: Optional[StrDict] = field(
        default_factory=lambda: {"type": "SpanTokensMasker", "arguments": [{"mask_prob": 0.1}]}
    )

    accelerator: Optional[str] = None

    # Output
    output_directory: str = "chemformer-pretrain"


cs = ConfigStore.instance()
cs.store(name="pretrain", node=Pretrain)
