from dataclasses import dataclass
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ExtractTemplates:
    """Config for aizynthmodels/template_based/tools/extract_templates.py"""

    input_data: str = MISSING
    output_data: str = MISSING
    radius: int = MISSING
    smiles_column: str = MISSING

    batch: Optional[Tuple[int, int]] = None
    expand_ring: bool = False
    expand_hetero: bool = False
    ringbreaker_column: str = ""


cs = ConfigStore.instance()
cs.store(name="extract_templates", node=ExtractTemplates)
