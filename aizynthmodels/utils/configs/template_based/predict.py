from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.template_based.core import CoreConfig


@dataclass
class Predict(CoreConfig):
    # Overriding from the base class
    unique_templates: Optional[str] = None  # File with a list of templates corresponding to the labels (.csv.gz)
    dataset_part: str = "test"

    # Output files
    output_predictions: str = "metrics_scores.csv"


cs = ConfigStore.instance()
cs.store(name="predict", node=Predict)
