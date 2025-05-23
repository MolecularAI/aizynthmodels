from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.route_distances.core import CoreConfig
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class Predict(CoreConfig):
    dataset_part: str = "full"
    datamodule: Optional[StrDict] = field(default_factory=lambda: {"type": "TreeListDataModule", "arguments": []})

    output_predictions: str = "predictions.csv"


cs = ConfigStore.instance()
cs.store(name="predict", node=Predict)
