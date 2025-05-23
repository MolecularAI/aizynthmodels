from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseLogger:
    """Parameters shared between different pytorch loggers."""

    save_dir: str = "${output_directory}"
    name: str = "${task}"
    version: Optional[str] = None
    prefix: str = ""


@dataclass
class CSVLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.CSVLogger"


@dataclass
class TensorboardLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    log_graph: bool = False
    default_hp_metric: bool = True
