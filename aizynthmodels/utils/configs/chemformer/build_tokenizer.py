from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BuildTokenizer:
    data_path: str = MISSING  # tab-separated .csv file
    tokenizer_path: str = MISSING
    smiles_column: str = "canonical_smiles"
    mol_opt_tokens_path: str = "mol_opt_tokens.txt"
    prop_pred_tokens_path: str = "prop_pred_tokens.txt"
    num_unused_tokens: int = 200


cs = ConfigStore.instance()
cs.store(name="build_tokenizer", node=BuildTokenizer)
