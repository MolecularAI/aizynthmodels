from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.chemformer.core import CoreConfig


@dataclass
class InferenceScore(CoreConfig):

    # Overriding from the base class
    callbacks: List[Any] = field(default_factory=lambda: ["ScoreCallback"])
    scorers: List[Any] = field(
        default_factory=lambda: [
            "FractionInvalidScore",
            "FractionUniqueScore",
            {"TanimotoSimilarityScore": [{"statistics": "mean"}]},
            "TopKAccuracyScore",
        ]
    )

    # Data
    seed: int = 1
    n_chunks: int = 1  # Number of chunks to divide the data into (for parallel inference)
    i_chunk: int = 0  # The idx of the current chunk
    dataset_part: str = "test"  # Which dataset split to run inference on. [full", "train", "val", "test"]

    # Output
    output_score_data: str = "metrics_scores.csv"
    output_predictions: Optional[str] = None  # .json file


cs = ConfigStore.instance()
cs.store(name="inference_score", node=InferenceScore)
