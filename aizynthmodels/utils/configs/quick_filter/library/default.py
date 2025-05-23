from dataclasses import dataclass

@dataclass
class FilterLibraryColumns:
    reaction_smiles: str = "reaction_smiles"
    reaction_hash: str = "reaction_hash"
    label: str = "label"
