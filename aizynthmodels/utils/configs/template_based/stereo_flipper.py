from dataclasses import dataclass
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class StereoFlipper:
    """Config for aizynthmodels/template_based/tools/stereo_flipper.py"""

    input_data: str = MISSING
    query: str = MISSING
    output_data: str = MISSING

    batch: Optional[Tuple[int, int]] = None
    template_column: str = "RetroTemplate"
    template_hash_column: str = "TemplateHash"


cs = ConfigStore.instance()
cs.store(name="stereo_flipper", node=StereoFlipper)
