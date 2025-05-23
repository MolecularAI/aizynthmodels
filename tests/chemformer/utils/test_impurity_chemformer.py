import numpy as np
import pandas as pd
import pytest

from aizynthmodels.chemformer.data import SynthesisDataModule
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
from aizynthmodels.chemformer.utils.impurity_chemformer import (
    default_purification_agents,
    get_dimerization_data,
    get_overreaction_data,
    get_purification_data,
    get_reaction_components_data,
    get_solvent_interaction_data,
    setup_datamodule,
    unravel_impurity_predictions,
)


def test_purification_agents():
    agents = default_purification_agents("standard")
    assert all(agent in agents for agent in ["O", "CC#N"])

    agents = default_purification_agents("supercritical_fluid_ms")
    assert all(agent in agents for agent in ["O", "CC#N", "N", "C(=O)O"])

    with pytest.raises(
        ValueError,
        match="'experiment_mode' should be set to either 'standard' or 'supercritical_fluid_ms', not 'dummy_mode'.",
    ):
        default_purification_agents("dummy_mode")


def test_unravel_impurity_predictions():
    df_in = pd.DataFrame(
        {
            "reactants": ["CCO.CCO"],
            "target_smiles": ["CC=OCCO"],
            "predicted_impurity": [["CCOCCO", "CC=OCCO"]],
            "log_likelihood": [[-2.06, -2.07]],
            "mode": ["Dimerization"],
        }
    )

    df_out = unravel_impurity_predictions(df_in)

    df_expected = pd.DataFrame(
        {
            "reactants": ["CCO.CCO", "CCO.CCO"],
            "product_ground_truth": ["CC=OCCO", "CC=OCCO"],
            "predicted_impurity": ["CCOCCO", "CC=OCCO"],
            "top_k": [1, 2],
            "log_likelihood": [-2.06, -2.07],
            "mode": ["Dimerization", "Dimerization"],
        }
    )

    assert df_out.equals(df_expected)


@pytest.mark.parametrize(
    ("reactants", "products", "reagents", "model_reagents", "expected"),
    [
        (
            ["CCO", "CO"],
            ["CC(=O)CC"],
            ["O"],
            True,
            ["CCO.CCO.O", "CO.CO.O", "CC(=O)CC.CC(=O)CC.O"],
        ),
        (
            ["CCO", "CO"],
            ["CC(=O)CC"],
            ["O"],
            False,
            ["CCO.CCO", "CO.CO", "CC(=O)CC.CC(=O)CC"],
        ),
        (
            ["CCO", "CO"],
            ["CC(=O)CC"],
            [],
            True,
            ["CCO.CCO", "CO.CO", "CC(=O)CC.CC(=O)CC"],
        ),
    ],
)
def test_dimerization(reactants, products, reagents, model_reagents, expected):
    dimerization = get_dimerization_data(reactants, products, reagents, model_reagents)
    assert all(output == exp_out for output, exp_out in zip(dimerization.reactants.values, expected))
    assert all(dimerization["mode"].values == "Dimerization")


@pytest.mark.parametrize(
    ("reactants", "products", "reagents", "model_reagents", "expected"),
    [
        (
            ["CCO", "CO", "CC"],
            ["CC(=O)CC"],
            ["O"],
            True,
            pd.DataFrame(
                {
                    "reactants": [
                        "CC.CC(=O)CC.O",
                        "CCO.CC(=O)CC.O",
                        "CCO.CC.CC(=O)CC.O",
                        "CCO.CC.O",
                        "CCO.CO.CC(=O)CC.O",
                        "CCO.CO.CC.CC(=O)CC.O",
                        "CCO.CO.O",
                        "CO.CC(=O)CC.O",
                        "CO.CC.CC(=O)CC.O",
                        "CO.CC.O",
                    ],
                    "mode": [
                        "Over-reaction",
                        "Over-reaction",
                        "Over-reaction",
                        "Reactant subset",
                        "Over-reaction",
                        "Over-reaction",
                        "Reactant subset",
                        "Over-reaction",
                        "Over-reaction",
                        "Reactant subset",
                    ],
                }
            ),
        ),
        (
            ["CCO", "CO"],
            ["CC(=O)CC"],
            ["O"],
            False,
            pd.DataFrame(
                {
                    "reactants": ["CCO.CC(=O)CC", "CCO.CO.CC(=O)CC", "CO.CC(=O)CC"],
                    "mode": ["Over-reaction", "Over-reaction", "Over-reaction"],
                }
            ),
        ),
        (
            ["CCO", "CO"],
            ["CC(=O)CC"],
            [],
            True,
            pd.DataFrame(
                {
                    "reactants": ["CCO.CC(=O)CC", "CCO.CO.CC(=O)CC", "CO.CC(=O)CC"],
                    "mode": ["Over-reaction", "Over-reaction", "Over-reaction"],
                }
            ),
        ),
    ],
)
def test_overreaction_data(reactants, products, reagents, model_reagents, expected):
    data = get_overreaction_data(reactants, products, reagents, model_reagents)
    data = data.sort_values(by="reactants", ignore_index=True)
    assert data.equals(expected)


def test_get_purification_data():
    data = get_purification_data(["CCOCC", "CC(=O)CC"], ["O", "CO"])
    expected = pd.DataFrame(
        {
            "reactants": ["CCOCC.O", "CCOCC.CO", "CC(=O)CC.O", "CC(=O)CC.CO"],
            "mode": [
                "Purification step reaction",
                "Purification step reaction",
                "Purification step reaction",
                "Purification step reaction",
            ],
        }
    )
    assert data.equals(expected)

    data = get_purification_data(["CCOCC", "CC(=O)CC"], [])
    assert data.empty


def test_get_reaction_component_data():
    data = get_reaction_components_data(["CCOCC", "CC(=O)CC"])
    expected = pd.DataFrame(
        {
            "reactants": ["-", "-"],
            "mode": ["Reaction component", "Reaction component"],
            "predicted_impurity": [["CCOCC"], ["CC(=O)CC"]],
            "log_likelihood": [[np.nan], [np.nan]],
        }
    )
    assert data.equals(expected)


def test_get_solvent_interaction_data():
    # solvent + model_reagents=True
    data = get_solvent_interaction_data(["CCO", "CC"], ["CCOCC", "CC(=O)CC"], ["O"], ["CO"], True)
    expected = pd.DataFrame(
        {
            "reactants": ["CCO.O.CO", "CC.O.CO", "CCOCC.O.CO", "CC(=O)CC.O.CO"],
            "mode": [
                "Solvent interaction",
                "Solvent interaction",
                "Solvent interaction",
                "Solvent interaction",
            ],
        }
    )

    assert data.equals(expected)

    # solvent + model_reagents=False
    data = get_solvent_interaction_data(["CCO", "CC"], ["CCOCC", "CC(=O)CC"], ["O"], ["CO"], False)
    expected = pd.DataFrame(
        {
            "reactants": ["CCO.O", "CC.O", "CCOCC.O", "CC(=O)CC.O"],
            "mode": [
                "Solvent interaction",
                "Solvent interaction",
                "Solvent interaction",
                "Solvent interaction",
            ],
        }
    )

    assert data.equals(expected)

    # no solvent
    data = get_solvent_interaction_data(["CCO", "CC"], ["CCOCC", "CC(=O)CC"], [], ["CO"], False)
    assert data.empty


def test_setup_datamodule():

    tokenizer = ChemformerTokenizer()
    smiles_list = ["CCO", "CC"]

    datamodule = setup_datamodule(smiles_list, tokenizer, 3)
    assert isinstance(datamodule, SynthesisDataModule)
    assert len(datamodule.full_dataloader()) == 1

    datamodule = setup_datamodule(smiles_list, tokenizer, 1)
    assert isinstance(datamodule, SynthesisDataModule)
    assert len(datamodule.full_dataloader()) == 2
