import numpy as np
import pandas as pd
import pytest
from rxnutils.chem.utils import remove_atom_mapping, split_rsmi
from scipy import sparse


@pytest.fixture
def model_hyperparams():
    params = {
        "num_features": 3,
        "num_hidden_layers": 2,
        "num_hidden_nodes": 2,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    return params


@pytest.fixture
def dummy_data():
    input_data = np.asarray(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
    )
    input_mat = sparse.lil_matrix(input_data).tocsr()

    labels_data = np.asarray(
        [
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
        ]
    )
    return input_mat, labels_data


@pytest.fixture
def reaction_data():
    reactants = [
        "CCO.CO",
        "CC.O",
        "CCO",
        "CS.O",
        "CSO.CC",
        "CCSO.C(=O)C",
        "CC(=O)C.CS",
        "CCOCC.CO",
        "CC(=O)CSC.CO",
        "CO.CSOC",
    ]
    products = ["CC(=O)CO", "CCO", "CCO", "CSO", "CSCC", "CCSC(=O)C", "CC(=O)S", "CCOCCC", "CC(OCO)CSC", "COSOC"]
    return reactants, products


@pytest.fixture
def dummy_datafiles(dummy_data, reaction_data, tmpdir):
    inputs, labels = dummy_data

    sparse.save_npz(tmpdir / "dummy_training_inputs_prod.npz", inputs[:8, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_training_inputs_rxn.npz", inputs[:8, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_training_inputs.npz", inputs[:8, :], compressed=True)
    np.savez(tmpdir / "dummy_training_labels.npz", labels[:8])

    sparse.save_npz(tmpdir / "dummy_validation_inputs_prod.npz", inputs[8:9, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_validation_inputs_rxn.npz", inputs[8:9, :], compressed=True)
    np.savez(tmpdir / "dummy_validation_labels.npz", labels[8:9])

    sparse.save_npz(tmpdir / "dummy_testing_inputs_prod.npz", inputs[9:, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_testing_inputs_rxn.npz", inputs[9:, :], compressed=True)
    np.savez(tmpdir / "dummy_testing_labels.npz", labels[9:])

    reactions = pd.DataFrame(
        {
            "reaction_smiles": [
                f"{reactants}>>{products}" for reactants, products in zip(reaction_data[0], reaction_data[1])
            ],
            "reactants": reaction_data[0],
            "products": reaction_data[1],
        }
    )

    train_reactions = reactions.iloc[:8]
    train_reactions["set"] = "train"
    val_reactions = reactions.iloc[8:9]
    val_reactions["set"] = "val"
    test_reactions = reactions.iloc[9:]
    test_reactions["set"] = "test"

    test_reactions.to_csv(tmpdir / "dummy_testing_reactions.csv", sep="\t", index=False)

    full_data = pd.concat([train_reactions, val_reactions, test_reactions], axis=0, ignore_index=True)
    full_data.to_csv(tmpdir / "dummy_reactions.csv", sep="\t", index=False)
    return str(tmpdir / "dummy")
