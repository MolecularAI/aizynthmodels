from dataclasses import dataclass
from typing import Optional


@dataclass
class CoreTrainer:
    """Minimal trainer config needed to run inference."""

    _target_: str = "pytorch_lightning.Trainer"
    accelerator: str = "auto"
    devices: int = "${n_devices}"
    strategy: str = "auto"
    precision: str = "16-mixed"
    deterministic: bool = True


@dataclass
class Trainer(CoreTrainer):
    """Trainer config needed to run model training. Includes the parameters in
    CoreTrainer."""

    num_nodes: int = "${n_nodes}"
    min_epochs: int = "${n_epochs}"
    max_epochs: int = "${n_epochs}"
    accumulate_grad_batches: Optional[int] = "${acc_batches}"
    gradient_clip_val: float = "${clip_grad}"
    limit_val_batches: float = "${limit_val_batches}"
    check_val_every_n_epoch: int = "${check_val_every_n_epoch}"
    profiler: Optional[str] = None
    enable_progress_bar: bool = True
