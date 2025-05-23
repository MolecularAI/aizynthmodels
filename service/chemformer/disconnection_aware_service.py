import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import omegaconf as oc
import pandas as pd
from fastapi import FastAPI
from rxnutils.chem.disconnection_sites.tag_converting import convert_atom_map_tag
from rxnutils.chem.utils import remove_atom_mapping
from service_utils import get_predictions

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.chemformer.utils.disconnection_aware import (
    predictions_atom_mapping,
    propagate_input_mapping_to_reactants,
    tag_current_bond,
    verify_disconnection,
)
from aizynthmodels.utils.configs.chemformer.predict import Predict

app = FastAPI()

# Container for data, classes that can be loaded upon startup of the REST API

config = oc.OmegaConf.structured(Predict)

config.batch_size = 64

if "CHEMFORMER_MODEL" in os.environ:
    model_path_vanilla = os.environ["CHEMFORMER_MODEL"]
else:
    model_path_vanilla = None

config.model_path = os.environ["CHEMFORMER_DISCONNECTION_MODEL"]
config.sample_unique = True  # Make sure we output unique predictions

config.task = os.environ["CHEMFORMER_TASK"]

config.vocabulary_path = os.environ["CHEMFORMER_VOCAB"]
config.datamodule = None

MODELS = {
    "chemformer_disconnect": Chemformer(config),
}

if model_path_vanilla:
    config.model_path = model_path_vanilla
    MODELS["chemformer_vanilla"] = Chemformer(config)


def _get_n_predictions(predicted_reactants: List[List[str]]):
    return [len(smiles_list) for smiles_list in predicted_reactants]


def _reshape(smiles_list: List[str], n_predictions: List[int]):
    reshaped_smiles_list = []
    counter = 0
    for n_pred in n_predictions:
        all_predictions = [smiles for smiles in smiles_list[counter : counter + n_pred]]  # noqa: E203
        counter += n_pred
        reshaped_smiles_list.append(all_predictions)
    return reshaped_smiles_list


@app.post("/chemformer-disconnect-api/predict-disconnection")
def predict_disconnection(smiles_list: List[str], bonds_list: List[List[int]], n_predictions: int = 10) -> List[Dict]:
    """
    Make prediction with disconnection-Chemformer given list of input SMILES and
    corresponding list of bonds to break [one bond per input SMILES].
    Returns the basic predictions and input product (with new atom-mapping)
    for each bond in each product. Tailored to the multi-step disconnection
    approach in aizynthfinder.

    Args:
        smiles_list: batch of input SMILES to model
        bonds: list of bonds to break for each input SMILES (one bond per molecule)
        n_predictions: number of beams in beam search
    """
    # Get input SMILES to the prediction and tag SMILES using the corresponding bonds
    # for that input.
    smiles_atom_map_tagged = [
        tag_current_bond(smiles, bond_atom_inds) for smiles, bond_atom_inds in zip(smiles_list, bonds_list)
    ]

    smiles_tagged_list = [convert_atom_map_tag(smiles) for smiles in smiles_atom_map_tagged]

    output = []
    predicted_smiles, log_lhs, _ = get_predictions(MODELS["chemformer_disconnect"], smiles_tagged_list, n_predictions)
    n_predictions = _get_n_predictions(predicted_smiles)

    # Get atom-mapping of predicted reaction
    mapped_rxns, _ = predictions_atom_mapping(smiles_list, predicted_smiles)

    reactants_mapped = np.array([mapped_rxn.split(">")[0] for mapped_rxn in mapped_rxns])
    product_new_mapping = np.array([mapped_rxn.split(">")[-1] for mapped_rxn in mapped_rxns])

    output = []
    for item_pred, item_lhs, item_smiles, item_mapped_product, item_bond in zip(
        _reshape(reactants_mapped, n_predictions),
        log_lhs,
        smiles_list,
        _reshape(product_new_mapping, n_predictions),
        bonds_list,
    ):
        output.append(
            {
                "input": item_smiles,
                "output": list(item_pred),
                "lhs": [float(val) for val in item_lhs],
                "product_new_mapping": list(item_mapped_product),
                "current_bond": item_bond,
            }
        )

    return output


@app.post("/chemformer-disconnect-api/predict")
def predict(
    smiles_list: List[str],
    bonds_list: List[Sequence[Sequence[int]]],
    n_predictions: int = 10,
) -> List[Dict]:
    """
    Make prediction with disconnection-Chemformer given list of input SMILES and
    corresponding list of bonds to break [one list of bonds per input SMILES].
    Makes prediction with vanilla Chemformer when bonds to break for one molecule is
    missing. Propagates the input atom-mapping to the predicted reactants.
    Performs check if the prompted bond was broken.

    Args:
        smiles_list: batch of input SMILES to model
        bonds_list: list of bonds to break for each input SMILES
        n_beams: number of beams in beam search
    """

    if "chemformer_vanilla" not in MODELS:
        raise ValueError(
            "A vanilla chemformer model was not specified as an environment variable "
            "before launching the service: 'export CHEMFORMER_MODEL=/path/to/chemformer/model.ckpt'"
        )

    # Get input SMILES to the prediction and tag SMILES if there are any bonds
    # prompts for that input. Create separate lists for disconnection-Chemformer model
    # and vanilla Chemformer.
    smiles_tagged_list = []
    smiles_input_atom_mapped = []
    smiles_untagged_list = []
    bond_atom_inds = []
    bonds_to_break = []
    for smiles, bonds in zip(smiles_list, bonds_list):
        if len(bonds) > 0:
            current_bond_inds = bonds.pop(0)
            smiles_input_atom_mapped.append(smiles)

            smiles_atom_map_tagged = tag_current_bond(smiles, current_bond_inds)
            smiles_tagged = convert_atom_map_tag(smiles_atom_map_tagged)

            smiles_tagged_list.append(smiles_tagged)
            bond_atom_inds.append(current_bond_inds)
            bonds_to_break.append(bonds)
        else:
            smiles_untagged_list.append(remove_atom_mapping(smiles))

    output = []
    if smiles_tagged_list:
        predicted_smiles, log_lhs, _ = get_predictions(
            MODELS["chemformer_disconnect"], smiles_tagged_list, n_predictions
        )
        n_predictions = _get_n_predictions(predicted_smiles)

        # Get atom-mapping of predicted reaction
        mapped_rxns, atom_map_confidences = predictions_atom_mapping(smiles_input_atom_mapped, predicted_smiles)

        reactants_mapped = np.array([mapped_rxn.split(">")[0] for mapped_rxn in mapped_rxns])
        product_new_mapping = np.array([mapped_rxn.split(">")[-1] for mapped_rxn in mapped_rxns])

        data = pd.DataFrame(
            {
                "input_smiles": smiles_input_atom_mapped,
                "product_new_mapping": list(_reshape(product_new_mapping, n_predictions)),
                "reactants_mapped": list(_reshape(reactants_mapped, n_predictions)),
                "log_lhs": list(log_lhs),
                "mapped_rxn": list(_reshape(mapped_rxns, n_predictions)),
                "confidence": list(_reshape(atom_map_confidences, n_predictions)),
                "bond_atom_inds": bond_atom_inds,
                "bonds_to_break": bonds_to_break,
            }
        )

        # Convert RXN-mapper atom-mapping of the reactant to the input product
        # atom-mapping.
        for i, row in data.iterrows():
            predictions_mapped = []
            is_bond_broken = []
            for prod_new_mapping, reactants_smiles in zip(row.product_new_mapping, row.reactants_mapped):
                mapped_reactants_smiles = propagate_input_mapping_to_reactants(
                    row.input_smiles, reactants_smiles, prod_new_mapping
                )

                # Verify that the prompted bond was actually disconnected
                is_disconnected = verify_disconnection(mapped_reactants_smiles, row.bond_atom_inds)

                is_bond_broken.append(is_disconnected)
                predictions_mapped.append(mapped_reactants_smiles)

            output.append(
                _get_output_dict(
                    row.input_smiles,
                    predictions_mapped,
                    row.log_lhs,
                    list(row.product_new_mapping),
                    list(row.mapped_rxn),
                    list(row.confidence),
                    row.bond_atom_inds,
                    row.bonds_to_break,
                    is_bond_broken,
                )
            )

    if smiles_untagged_list:
        # Get predictions from vanilla Chemformer predictions if the list of bonds is
        # empty
        (predicted_smiles_untagged, log_lhs_untagged, _) = get_predictions(
            MODELS["chemformer_vanilla"], smiles_untagged_list, n_predictions
        )

        output_untagged = _convert_output(predicted_smiles_untagged, log_lhs_untagged, smiles_untagged_list)

        output.extend(output_untagged)

    return output


def _get_output_dict(
    smiles: str,
    predicted_smiles: List[str],
    log_lhs: List[float],
    mapped_product: List[str],
    mapped_rxn: List[str],
    confidence: List[float],
    current_bond: List[int],
    bonds_to_break: List[List[int]],
    is_disconnected: List[bool],
) -> Dict[str, Any]:
    output = {
        "input": smiles,
        "output": predicted_smiles,
        "lhs": [float(val) for val in log_lhs],
        "product_new_mapping": mapped_product,
        "mapped_rxn": mapped_rxn,
        "confidence": confidence,
        "current_bond": current_bond,
        "bonds_to_break": bonds_to_break,
        "is_disconnected": is_disconnected,
    }
    return output


def _convert_output(
    predicted_smiles: List[str],
    log_lhs: List[float],
    original_smiles: str,
    mapped_product: List[str] = [],
    mapped_rxn: List[str] = [],
    confidence: List[float] = [],
    current_bond: List[int] = [],
    bonds_to_break: List[List[int]] = [],
    is_disconnected: Optional[bool] = None,
):
    """Convert output to correct format."""
    output = []
    for item_pred, item_lhs, item_smiles in zip(predicted_smiles, log_lhs, original_smiles):
        output.append(
            _get_output_dict(
                item_smiles,
                list(item_pred),
                item_lhs,
                mapped_product,
                mapped_rxn,
                confidence,
                current_bond,
                bonds_to_break,
                is_disconnected,
            )
        )
    return output


if __name__ == "__main__":
    import uvicorn

    port = os.environ.get("PORT") or 8023

    uvicorn.run(
        "disconnection_aware_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
