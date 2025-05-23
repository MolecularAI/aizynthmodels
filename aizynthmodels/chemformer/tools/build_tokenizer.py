import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
from aizynthmodels.chemformer.utils.defaults import REGEX

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import build_tokenizer
from aizynthmodels.utils.hydra import custom_config


def read_extra_tokens(paths):
    extra_tokens = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            text = p.read_text()
            tokens = text.split("\n")
            tokens = [token for token in tokens if token != ""]
            logging.info(f"Read {len(tokens)} tokens from {path}")
            extra_tokens.extend(tokens)

    return extra_tokens


def build_unused_tokens(num_tokens):
    tokens = []
    for i in range(num_tokens):
        token = f"<UNUSED_{str(i)}>"
        tokens.append(token)

    return tokens


@hydra.main(version_base=None, config_name="build_tokenizer")
@custom_config
def main(config: DictConfig) -> None:
    logging.info("Reading molecule dataset...")
    mol_dataset = pd.read_csv(config.data_path, sep="\t")
    smiles = mol_dataset[config.smiles_column].values.tolist()
    logging.info("Completed reading dataset.")

    logging.info("Reading extra tokens...")
    paths = [config.mol_opt_tokens_path, config.prop_pred_tokens_path]
    extra_tokens = read_extra_tokens(paths)
    unused_tokens = build_unused_tokens(config.num_unused_tokens)
    logging.info("Completed reading extra tokens.")

    logging.info("Building tokenizer...")
    tokenizer = ChemformerTokenizer(
        smiles=smiles,
        tokens=extra_tokens + unused_tokens,
        regex_token_patterns=REGEX.split("|"),
    )
    logging.info("Completed building tokenizer.")

    logging.info("Writing tokenizer...")
    tokenizer.save_vocabulary(config.tokenizer_path)
    logging.info("Complete.")


if __name__ == "__main__":
    main()
