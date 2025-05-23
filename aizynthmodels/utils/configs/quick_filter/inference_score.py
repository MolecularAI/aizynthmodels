from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.quick_filter.core import CoreConfig
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class InferenceScore(CoreConfig):

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
        default_factory=lambda: ["BinaryAccuracyScore", "BalancedAccuracyScore", "AveragePrecisionScore", "RecallScore"]
    )

    # Output files
    output_predictions: str = "predictions.json"
    output_score_data: str = "metrics_scores.csv"


cs = ConfigStore.instance()
cs.store(name="inference_score", node=InferenceScore)
