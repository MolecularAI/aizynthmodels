"""
Module for generating "negative" reactions by applying
a random template in the forward direction and collecting reactions not identical
to any recorded reaction
"""

from typing import Dict

import pandas as pd
from omegaconf import DictConfig
from rdkit import Chem
from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.template import ReactionTemplate


def random_template_selection(
    base_row: pd.Series,
    name_config: DictConfig,
    full_df: pd.DataFrame,
    negative_data: Dict,
    nsamples: int = 1000,
    random_state: int = 1984,
) -> int:
    """
    Take a row from a template-library and randomly selects other rows from the template library
    from which templates are applied in the forward direction

    Return a new reaction record for each of the generated outcomes of the template
    application that does not correspond to the original reaction


    :param base_row: a row from a template library
    :param name_config: a configuration with column names
    :param negative_data: a dictionary to add the generate reactions to
    :returns: the number of generated data points
    """
    rxn = ChemicalReaction(base_row[name_config["reaction_smiles"]], clean_smiles=False)
    ref_mol = rxn.products[0]
    if not rxn.products:
        return 0

    nadded = 0
    selected_templates = set()
    nsamples = min(nsamples, len(full_df))
    for _, trial_row in full_df.sample(nsamples, replace=False, random_state=random_state).iterrows():
        if trial_row[name_config["template_hash"]] in selected_templates:
            continue
        if trial_row[name_config["template_hash"]] == base_row[name_config["template_hash"]]:
            continue

        smarts_fwd = ">>".join(trial_row[name_config["retro_template"]].split(">>")[::-1])
        try:
            new_product = ReactionTemplate(smarts_fwd).apply(rxn.reactants_smiles)[0][0]
        except (ValueError, IndexError):
            continue

        # Ignore stereochemistry in comparison
        new_product_hash = Chem.MolToInchiKey(Chem.MolFromSmiles(new_product))
        if Chem.MolToInchiKey(ref_mol)[:16] == new_product_hash[::16]:
            continue

        selected_templates.add(trial_row[name_config["template_hash"]])
        negative_data["reaction_smiles"].append(rxn.reactants_smiles + ">>" + new_product)
        negative_data["reaction_hash"].append(
            base_row[name_config["reaction_hash"]].split(">")[0] + ">>" + new_product_hash
        )
        negative_data["retro_template"].append(trial_row[name_config["retro_template"]])
        negative_data["template_hash"].append(trial_row[name_config["template_hash"]])
        negative_data["classification"].append("0.99")
        negative_data["ring_breaker"].append(False)
        nadded += 1
    return nadded
