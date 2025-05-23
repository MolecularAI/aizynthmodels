from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore


@dataclass
class PreprocessingPipeline:
    seed: int = 11
    selected_reactions_path: str = "selected_reactions.csv"
    reaction_components_path: str = "reaction_components.csv"
    routes_to_exclude: List[str] = field(default_factory=list)
    training_fraction: float = 0.9
    reaction_hash_col: str = "reaction_hash"
    set_col: str = "set"
    is_external_col: str = "is_external"


@dataclass
class TaggingPipeline:
    tagged_reaction_data_path: str = "proc_selected_reactions_disconnection.csv"
    autotag_data_path: str = "proc_selected_reactions_autotag.csv"


@dataclass
class DataPreprocessing:
    nbatches: int = 200
    chemformer_data_path: str = "proc_selected_reactions.csv"

    preprocessing_pipeline: PreprocessingPipeline = PreprocessingPipeline()
    tagging_pipeline: TaggingPipeline = TaggingPipeline()


cs = ConfigStore.instance()
cs.store(name="data_prep_pipeline", node=DataPreprocessing)
