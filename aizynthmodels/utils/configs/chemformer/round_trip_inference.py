from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from aizynthmodels.utils.configs.chemformer.inference_score import InferenceScore


@dataclass
class RoundTripInference(InferenceScore):
    input_data: str = MISSING  # The original input data to inference_score.py
    backward_predictions: str = MISSING  # The predictions (.json) from inference_score.py
    target_column: str = "products"  # Column with products that should be reproduced
    n_predictions: int = 1

    data_path: str = MISSING  # Placeholder arg -> set during inference

    scorers: List[Any] = field(
        default_factory=lambda: [
            "FractionInvalidScore",
            "FractionUniqueScore",
            "TopKAccuracyScore",
            "TopKCoverageScore",
        ]
    )


cs = ConfigStore.instance()
cs.store(name="round_trip_inference", node=RoundTripInference)
