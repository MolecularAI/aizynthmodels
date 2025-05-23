from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from aizynthmodels.utils.configs.quick_filter.train import Train
from aizynthmodels.utils.configs.quick_filter.generation import NegativeReactionGenerator
from aizynthmodels.utils.configs.quick_filter.library.default import FilterLibraryColumns
from aizynthmodels.utils.configs.template_based.library.default import TemplateLibraryColumns

@dataclass
class FilterModelPipeline(Train):
    """ Configuration for aizynthmodels/quick_filter/pipelines/filter_model_pipeline.py
    """
    negative_generation: NegativeReactionGenerator = field(default_factory=NegativeReactionGenerator)
    library_columns: FilterLibraryColumns = field(default_factory=FilterLibraryColumns)
    template_library_columns: TemplateLibraryColumns = field(default_factory=TemplateLibraryColumns)

    python_kernel: str = MISSING
    n_batches: int = 200
    training_fraction: float = 0.9
    onnx_model: str = MISSING

    batch: Optional[List[int]] = None  # Is set by the pipeline itself


cs = ConfigStore.instance()
cs.store(name="filter_pipeline", node=FilterModelPipeline)
