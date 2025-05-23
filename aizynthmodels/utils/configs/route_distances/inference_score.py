from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.route_distances.core import CoreConfig


@dataclass
class InferenceScore(CoreConfig):
    callbacks: List[Any] = field(default_factory=lambda: ["ScoreCallback"])
    scorers: List[Any] = field(
        default_factory=lambda: [
            "R2Score",
            "MeanAbsoluteError",
        ]
    )

    output_score_data: str = "metrics_scores.csv"
    output_predictions: Optional[str] = None  # .json file


cs = ConfigStore.instance()
cs.store(name="inference_score", node=InferenceScore)
