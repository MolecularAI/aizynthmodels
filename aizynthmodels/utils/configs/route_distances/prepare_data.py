from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class PrepareData:
    datapath: str = MISSING
    output: str = MISSING  # .pkl file

    fp_size: int = 2048
    use_reduced: bool = False


cs = ConfigStore.instance()
cs.store(name="prepare_data", node=PrepareData)
