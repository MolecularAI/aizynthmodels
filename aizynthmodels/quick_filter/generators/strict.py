"""
Module for generating "negative" reactions by applying
a template in the forward direction and collecting reactions not identical
to the recorded reaction for that template
"""

from typing import Dict

import pandas as pd
from omegaconf import DictConfig
from rdkit import Chem, RDLogger
from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.template import ReactionTemplate

rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


def strict_template_selection(base_row: pd.Series, name_config: DictConfig, negative_data: Dict, **_) -> int:
    """
    Take a row from a template-library and apply the template in the forward direction.

    Return a new reaction record for each of the generated outcomes of the template
    application that does not correspond to the original reaction


    :param base_row: a row from a template library
    :param name_config: a configuration with column names
    :param negative_data: a dictionary to add the generate reactions to
    :returns: the number of generated data points
    """
    rxn = ChemicalReaction(base_row[name_config["reaction_smiles"]], clean_smiles=False)
    if not rxn.products:
        return 0

    ref_mol = rxn.products[0]
    ref_inchi = Chem.MolToInchiKey(ref_mol)

    smarts_fwd = ">>".join(base_row[name_config["retro_template"]].split(">>")[::-1])
    new_products = ReactionTemplate(smarts_fwd).apply(rxn.reactants_smiles)

    nadded = 0
    for smiles_list in new_products:
        if not smiles_list:
            continue
        smiles = smiles_list[0]
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        inchi = Chem.MolToInchiKey(mol)
        # Ignore stereochemistry in comparison
        if inchi[:16] == ref_inchi[:16]:
            continue

        negative_data["reaction_smiles"].append(rxn.reactants_smiles + ">>" + smiles)
        negative_data["reaction_hash"].append(base_row[name_config["reaction_hash"]].split(">")[0] + ">>" + inchi)
        negative_data["retro_template"].append(base_row[name_config["retro_template"]])
        negative_data["template_hash"].append(base_row[name_config["template_hash"]])
        if name_config["classification"] in base_row.index:
            negative_data["classification"].append(base_row[name_config["classification"]])
        if name_config["ring_breaker"] in base_row.index:
            negative_data["ring_breaker"].append(base_row[name_config["ring_breaker"]])
        nadded += 1
    return nadded
