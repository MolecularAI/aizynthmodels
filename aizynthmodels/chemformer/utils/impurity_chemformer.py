from typing import List

import numpy as np
import pandas as pd

from aizynthmodels.chemformer.data import SynthesisDataModule
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
from aizynthmodels.chemformer.utils.defaults import DEFAULT_MAX_SEQ_LEN
from aizynthmodels.utils.smiles import canonicalize_smiles, enumerate_smiles_combinations


def default_purification_agents(experiment_mode: str = "standard") -> List[str]:
    """Returns a list of the purification agents typically used in standard or SPF-MS."""
    agents = ["O", "CC#N"]
    if experiment_mode == "supercritical_fluid_ms":
        agents += ["N", "C(=O)O"]
    elif experiment_mode != "standard":
        raise ValueError(
            f"'experiment_mode' should be set to either 'standard' or "
            f"'supercritical_fluid_ms', not '{experiment_mode}'."
        )
    return agents


def _create_input_dataframe(reacting_smiles: List[str], mode: str) -> pd.DataFrame:
    """Returns dataframe used for impurity prediction."""
    data = pd.DataFrame({"reactants": reacting_smiles, "mode": mode})
    return data


def get_dimerization_data(
    reactants: List[str], products: List[str], reagents: List[str], model_reagents: bool
) -> pd.DataFrame:
    """
    Returns dimers from reactants and products.
    """
    monomer_list = [*reactants, *products]

    if not model_reagents:
        reagents = []

    dimer_smiles = [".".join([monomer, monomer] + reagents) for monomer in monomer_list]
    return _create_input_dataframe(dimer_smiles, "Dimerization")


def get_overreaction_data(
    reactants: List[str], products: List[str], reagents: List[str], model_reagents: bool
) -> pd.DataFrame:
    """Returns the over-reaction and reactant-subset reactants"""
    molecule_combinations = enumerate_smiles_combinations([*reactants, *products], 2, len(reactants) + 1)

    if not model_reagents:
        reagents = []

    combinations_with_reagents = [".".join(combination + reagents) for combination in molecule_combinations]

    mode = []
    overreaction_smiles = []
    for input_smiles in combinations_with_reagents:
        if any(product in input_smiles for product in products):
            mode.append("Over-reaction")
            overreaction_smiles.append(input_smiles)
        elif len(input_smiles.split(".")) < len(reactants + reagents):
            mode.append("Reactant subset")
            overreaction_smiles.append(input_smiles)

    return _create_input_dataframe(overreaction_smiles, mode)


def get_purification_data(products: List[str], agents: List[str]) -> pd.DataFrame:
    """Returns the input to purification impurity predictions."""
    if not agents:
        return pd.DataFrame()

    reactants = [f"{product}.{agent}" for product in products for agent in agents]
    return _create_input_dataframe(reactants, "Purification step reaction")


def get_reaction_components_data(components: List[str]) -> pd.DataFrame:
    """Create a 'prediction results' dataframe for the reaction components."""
    reaction_components = pd.DataFrame(
        {
            "reactants": ["-"] * len(components),
            "mode": ["Reaction component"] * len(components),
            "predicted_impurity": [[component] for component in components],
            "log_likelihood": [[np.nan]] * len(components),
        }
    )
    return reaction_components


def get_solvent_interaction_data(
    reactants: List[str],
    impurity_products: List[str],
    solvent: List[str],
    reagents: List[str],
    model_reagents: bool,
) -> pd.DataFrame:
    """Returns the prediction input for solvent interaction impurity prediction."""
    if not solvent:
        return pd.DataFrame()

    if not model_reagents:
        reagents = []

    smiles_to_interact = reactants + impurity_products

    reactants = [".".join([smiles, sol_smiles] + reagents) for smiles in smiles_to_interact for sol_smiles in solvent]
    return _create_input_dataframe(reactants, "Solvent interaction")


def setup_datamodule(smiles: List[str], tokenizer: ChemformerTokenizer, batch_size: int):
    """Given a list of SMILES, set the datamodule used for generating input
    data to the Chemformer model.

    :param smiles: list of input SMILES
    :return: datamodule object which produces input data for Chemformer."""
    datamodule = SynthesisDataModule(
        dataset_path="",
        reactants=smiles,
        products=smiles,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
    )
    datamodule.setup()
    return datamodule


def unravel_impurity_predictions(impurities_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the impurity dataframe to list each (top-k) prediction rather than
    listing lists of predictions.

    :param impurities_raw: raw prediction data from Chemformer predictions.
    :return: dataframe with listed predictions.
    """
    predictions = []
    log_likelihoods = []
    top_k = []
    modes = []
    reactants = []
    products = []

    for _, row in impurities_raw.iterrows():
        counter = 1
        for prediction, log_lhs in zip(row.predicted_impurity, row.log_likelihood):
            predictions.append(prediction)
            log_likelihoods.append(log_lhs)
            top_k.append(counter)
            modes.append(row["mode"])
            reactants.append(row.reactants)
            products.append(row.target_smiles)

            counter += 1

    impurities = pd.DataFrame(
        {
            "reactants": reactants,
            "product_ground_truth": products,
            "predicted_impurity": predictions,
            "top_k": top_k,
            "log_likelihood": log_likelihoods,
            "mode": modes,
        }
    )

    impurities["predicted_impurity"] = impurities.apply(
        lambda row: canonicalize_smiles(row["predicted_impurity"]),
        axis=1,
    )
    return impurities.drop_duplicates(subset=["predicted_impurity"], keep="first", ignore_index=True)
