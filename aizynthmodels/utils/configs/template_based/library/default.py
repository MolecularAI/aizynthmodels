from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TemplateLibraryColumns:
    reaction_smiles: str = "reaction_smiles"
    reaction_hash: str = "reaction_hash"
    retro_template: str = "retro_template"
    template_hash: str = "template_hash"
    template_code: str = "template_code"
    library_occurrence: str = "library_occurence"
    classification: Optional[str] = "classification"
    ring_breaker: Optional[str] = "ring_breaker"
    stereo_bucket: Optional[str] = "stereo_bucket"
    flipped_stereo: Optional[str] = "flipped_stereo"


@dataclass
class LibraryConfig:
    columns: TemplateLibraryColumns = TemplateLibraryColumns()

    metadata_columns: List[str] = field(default_factory=lambda: ["template_hash", "classification"])
    template_set: str = "templates"
