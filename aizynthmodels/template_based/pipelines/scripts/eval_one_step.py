""" Module routines for evaluating one-step retrosynthesis model
"""

import json
import logging
import math
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pandas as pd
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rxnutils.chem.utils import split_rsmi, split_smiles_from_reaction

from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.hydra import custom_config

YAML_TEMPLATE = """expansion:
  default:
      - {}
      - {}
"""


def _create_config(model_path: str, templates_path: str) -> str:
    _, filename = tempfile.mkstemp(suffix=".yaml")
    with open(filename, "w") as fileobj:
        fileobj.write(YAML_TEMPLATE.format(model_path, templates_path))
    return filename


def _create_test_reaction(config: DictConfig) -> str:
    """
    Selected reactions for testing. Group reactions by parent reaction classification
    and then selected random reactions from each group.
    """
    columns = config.library_config.columns

    def transform_rsmi(row):
        reactants, _, products = split_rsmi(row[columns.reaction_smiles])
        rxn = AllChem.ReactionFromSmarts(f"{reactants}>>{products}")
        AllChem.RemoveMappingNumbersFromReactions(rxn)
        return AllChem.ReactionToSmiles(rxn)

    test_lib_path = get_filename(config, "library", "testing")
    data = pd.read_csv(test_lib_path, sep="\t")
    if columns.ring_breaker not in data.columns:
        data[columns.ring_breaker] = [False] * len(data)
    trunc_class = data[columns.classification].apply(lambda x: ".".join(x.split(" ")[0].split(".")[:2]))

    class_to_idx = defaultdict(list)
    for idx, val in enumerate(trunc_class):
        class_to_idx[val].append(idx)
    n_per_class = math.ceil(config.model_eval.n_test_reactions / len(class_to_idx))

    random.seed(1789)
    selected_idx = []
    for indices in class_to_idx.values():
        if len(indices) > n_per_class:
            selected_idx.extend(random.sample(indices, k=n_per_class))
        else:
            selected_idx.extend(indices)

    data_sel = data.iloc[selected_idx]
    rsmi = data_sel.apply(transform_rsmi, axis=1)
    filename = test_lib_path.replace(".csv", "_selected.csv")
    pd.DataFrame(
        {
            columns.reaction_smiles: rsmi,
            columns.ring_breaker: data_sel[columns.ring_breaker],
            "original_index": selected_idx,
        }
    ).to_csv(filename, index=False, sep="\t")
    return filename


def _eval_expander(
    expander_output: List[Dict[str, Any]],
    ref_reactions_path: str,
    config: DictConfig,
) -> None:
    columns = config.library_config.columns
    ref_reactions = pd.read_csv(ref_reactions_path, sep="\t")

    stats = defaultdict(list)
    for (_, row), output in zip(ref_reactions.iterrows(), expander_output):
        reactants, _, product = split_rsmi(row[columns.reaction_smiles])
        nrings_prod = CalcNumRings(Chem.MolFromSmiles(product))
        reactants_inchis = set(
            Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in split_smiles_from_reaction(reactants)
        )
        found = False
        found_idx = None
        ring_broken = False
        for idx, outcome in enumerate(output["outcome"]):
            outcome_inchis = set(
                Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in split_smiles_from_reaction(outcome)
            )
            if outcome_inchis == reactants_inchis:
                found = True
                found_idx = idx + 1
            nrings_reactants = sum(CalcNumRings(Chem.MolFromSmiles(smi)) for smi in split_smiles_from_reaction(outcome))
            if nrings_reactants < nrings_prod:
                ring_broken = True
        stats["target"].append(product)
        stats["expected reactants"].append(reactants)
        stats["found expected"].append(found)
        stats["rank of expected"].append(found_idx)
        if row[columns.ring_breaker]:
            stats["ring broken"].append(ring_broken)
        else:
            stats["ring broken"].append(None)
        stats["ring breaking"].append(bool(row[columns.ring_breaker]))
        stats["non-applicable"].append(output["non-applicable"])

    with open(get_filename(config, "onestep_report"), "w") as fileobj:
        json.dump(stats, fileobj)

    stats = pd.DataFrame(stats)
    logging.info(f"Evaluated {len(ref_reactions)} reactions")
    logging.info(f"Average found expected: {stats['found expected'].mean()*100:.2f}%")
    logging.info(f"Average rank of expected: {stats['rank of expected'].mean():.2f}")
    logging.info(f"Average ring broken when expected: {stats['ring broken'].mean()*100:.2f}%")
    logging.info(f"Percentage of ring reactions: {stats['ring breaking'].mean()*100:.2f}%")
    logging.info(f"Average non-applicable (in top-50): {stats['non-applicable'].mean():.2f}")


def _run_expander(ref_reactions_path: str, config_path: str, config: DictConfig) -> List[Dict[str, Any]]:
    input_arguments = [
        "conda",
        "run",
        "-p",
        config.model_eval.aizynthfinder_env,
        "python",
        str(Path(__file__).parent.parent.parent / "helpers" / "expander.py"),
        ref_reactions_path,
        config.library_config.columns.reaction_smiles,
        str(config.model_eval.top_n),
        config_path,
        get_filename(config, "expander_output"),
    ]
    subprocess.check_call(input_arguments)

    with open(get_filename(config, "expander_output"), "r") as fileobj:
        return json.load(fileobj)


@hydra.main(version_base=None, config_name="expansion_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    if "test_library" in config:
        ref_reactions_path = config.test_library
    else:
        ref_reactions_path = _create_test_reaction(config)

    if "model_path" in config and "templates_path" in config:
        config_path = _create_config(config.model_path, config.templates_path)
    else:
        config_path = _create_config(get_filename(config, "onnx_model"), get_filename(config, "unique_templates"))

    expander_output = _run_expander(ref_reactions_path, config_path, config)

    _eval_expander(expander_output, ref_reactions_path, config)


if __name__ == "__main__":
    main()
