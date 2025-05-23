from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.template_based.core import CoreConfig
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class InferenceScore(CoreConfig):

    # Overriding from the base class
    unique_templates: Optional[str] = None  # File with a list of templates corresponding to the labels (.csv.gz)
    group_on_templates: bool = True

    dataset_part: str = "test"

    datamodule: Optional[StrDict] = field(
        default_factory=lambda: {
            "type": "SmilesBasedDataModule",
            "arguments": [
                {"dataset_path": "${smiles_data_path}"},
                {"fingerprint_radius": "${model_hyperparams.fingerprint_radius}"},
                {"fingerprint_size": "${model_hyperparams.fingerprint_size}"},
                {"chirality": "${model_hyperparams.chirality}"},
            ],
        }
    )

    callbacks: List[Any] = field(default_factory=lambda: ["ScoreCallback"])

    scorers: List[Any] = field(
        default_factory=lambda: ["TopKAccuracyScore", "FractionUniqueScore", "FractionInvalidScore"]
    )

    # Output files
    output_score_data: str = "predictions.json"
    output_predictions: str = "metrics_scores.csv"


cs = ConfigStore.instance()
cs.store(name="inference_score", node=InferenceScore)
