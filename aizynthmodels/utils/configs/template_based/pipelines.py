from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from aizynthmodels.utils.configs.template_based.library import LibraryConfig, TemplateLibraryColumns
from aizynthmodels.utils.configs.template_based.train import Train
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class TrainingPipeline:
    python_kernel: str = MISSING
    training_output: str = "${output_directory}"
    n_batches: int = 200
    training_fraction: float = 0.9
    random_seed: int = "${random_seed}"
    selected_ids_path: Optional[str] = "selected_reactions_ids.json"
    routes_to_exclude: List[str] = field(default_factory=list)
    onnx_model: str = MISSING
    acc_batches: int = "${acc_batches}"
    check_val_every_n_epoch: int = "${check_val_every_n_epoch}"
    n_devices: int = "${n_devices}"
    n_nodes: int = "${n_nodes}"

    batch: Optional[List[int]] = None  # Is set by the pipeline itself


@dataclass
class ModelEvaluationPipeline:
    stock_for_finding: str = ""
    stock_for_recovery: str = ""
    search_properties_for_finding: StrDict = field(default_factory=lambda: {"return_first": True})
    search_properties_for_recovery: StrDict = field(
        default_factory=lambda: {"max_transforms": 10, "iteration_limit": 500, "time_limit": 3600}
    )
    reference_routes: str = ""
    target_smiles: str = ""
    top_n: int = 50
    n_test_reactions: int = 1000
    distance_model: Optional[str] = None
    aizynthfinder_env: Optional[str] = None


@dataclass
class ExpansionModelPipeline(Train):
    """ Configuration for aizynthmodels/template_based/pipelines/expansion_model_pipeline.py
    """
    library_config: LibraryConfig = LibraryConfig()
    training_pipeline: TrainingPipeline = TrainingPipeline()
    model_eval: ModelEvaluationPipeline = ModelEvaluationPipeline()


@dataclass
class TemplatePipeline:
    """Configuration for aizynthmodels/template_based/pipelines/template_pipeline.py
    """

    python_kernel: str
    data_import_class: str
    data_import_config: StrDict = field(default_factory=dict)
    selected_templates_prefix: str = ""
    selected_templates_postfix: str = "template_library.csv"
    import_data_path: str = "imported_reactions.csv"
    validated_reactions_path: str = "validated_reactions.csv"
    selected_reactions_path: str = "selected_reactions.csv"
    reaction_report_path: str = "reaction_selection_report.html"
    stereo_reactions_path: str = "stereo_reactions.csv"
    selected_stereo_reactions_path: str = "selected_stereo_reactions.csv"
    stereo_report_path: str = "stereo_selection_report.html"
    unvalidated_templates_path: str = "reaction_templates_unvalidated.csv"
    validated_templates_path: str = "reaction_templates_validated.csv"
    templates_report_path: str = "template_selection_report.html"
    min_template_occurrence: int = 10
    n_batches: int = 200
    template_library_columns: TemplateLibraryColumns = TemplateLibraryColumns()


cs = ConfigStore.instance()
cs.store(name="template_pipeline", node=TemplatePipeline)
cs.store(name="expansion_pipeline", node=ExpansionModelPipeline)
