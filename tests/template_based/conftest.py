import numpy as np
import pandas as pd
import pytest
from rxnutils.chem.utils import remove_atom_mapping, split_rsmi
from scipy import sparse


@pytest.fixture
def model_hyperparams():
    params = {
        "num_features": 3,
        "num_classes": 2,
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
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    labels_mat = sparse.lil_matrix(labels_data).tocsr()
    return input_mat, labels_mat


@pytest.fixture
def reaction_data(shared_datadir):

    data = pd.read_csv(shared_datadir / "dummy_template_library.csv", sep="\t")

    reaction_smiles = data.reaction_smiles.values
    rxn_components = [split_rsmi(rxn) for rxn in reaction_smiles]
    reactants = [remove_atom_mapping(components[0]) for components in rxn_components]
    products = [remove_atom_mapping(components[-1]) for components in rxn_components]

    templates = []
    template_hash = []
    for tp, th in zip(data.retro_template.values, data.template_hash.values):
        if th not in template_hash:
            template_hash.append(th)
            templates.append(tp)

    return reactants, products, templates


@pytest.fixture
def dummy_datafiles(dummy_data, reaction_data, tmpdir):
    inputs, labels = dummy_data

    sparse.save_npz(tmpdir / "dummy_training_inputs.npz", inputs[:8, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_training_labels.npz", labels[:8, :], compressed=True)

    sparse.save_npz(tmpdir / "dummy_validation_inputs.npz", inputs[8:9, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_validation_labels.npz", labels[8:9, :], compressed=True)

    sparse.save_npz(tmpdir / "dummy_testing_inputs.npz", inputs[9:, :], compressed=True)
    sparse.save_npz(tmpdir / "dummy_testing_labels.npz", labels[9:, :], compressed=True)

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
