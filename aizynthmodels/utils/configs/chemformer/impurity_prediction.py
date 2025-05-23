from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.chemformer.core import CoreConfig
from aizynthmodels.utils.configs.chemformer.model_hyperparams import ClientModelHyperparams
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class PredictImpurities(CoreConfig):
    # Overriding default config
    datamodule: Optional[List[Any]] = None

    # Trainer
    seed: int = 1

    n_chunks: int = 1  # Number of chunks to divide the data into (for parallel inference)
    i_chunk: int = 0  # The idx of the current chunk

    # Data
    dataset_part: str = "test"  # Which dataset split to run inference on. [full", "train", "val", "test"]

    # Model args
    model_reagents: bool = False
    n_predictions_baseline: int = 10
    n_predictions_non_baseline: int = 3
    n_predictions_purification: int = 1
    top_k_products: int = 3

    # Output
    output_predictions: str = "predicted_impurities.csv"

    # Fixed arguments for impurity prediction (do not change)
    task: str = "forward_prediction"
    sample_unique: bool = True

    n_predictions: Optional[int] = None  # Placeholder arg -> updated by ImpurityChemformer


@dataclass
class PredictImpuritiesClient(PredictImpurities):
    model_hyperparams: ClientModelHyperparams = ClientModelHyperparams()
    model: StrDict = field(
        default_factory=lambda: {
            "type": "ChemformerClient",
        }
    )


cs = ConfigStore.instance()
cs.store(name="predict_impurities", node=PredictImpurities)
cs.store(name="predict_impurities_client", node=PredictImpuritiesClient)
