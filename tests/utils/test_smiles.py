import numpy as np

from aizynthmodels.utils.smiles import (
    canonicalize_mapped_smiles,
    construct_reaction_smiles,
    enumerate_smiles_combinations,
    inchi_key,
    smiles_to_fingerprint,
    uniqueify_sampled_smiles,
)


def test_construct_reaction_smiles():
    product_smiles = "Clc1ccccc1"
    reactants_smiles = np.array(["c1ccccc1.Cl", "c1ccccc1.Br"])
    rxns = construct_reaction_smiles(product_smiles, reactants_smiles)
    assert len(rxns) == 2
    assert rxns[0] == "c1ccccc1.Cl>>Clc1ccccc1"
    assert rxns[1] == "c1ccccc1.Br>>Clc1ccccc1"


def test_canonicalize_mapped_smiles():
    input_smiles = "[cH:1]1[c:2]([Cl:3])[cH:4][cH:5][cH:6][cH:7]1"
    expected_smiles = "[Cl:3][c:2]1[cH:4][cH:5][cH:6][cH:7][cH:1]1"
    assert canonicalize_mapped_smiles(input_smiles) == expected_smiles

    input_smiles = "c1c(Cl)cccc1"
    expected_smiles = "Clc1ccccc1"
    assert canonicalize_mapped_smiles(input_smiles) == expected_smiles

    input_smiles = "not-a-smiles"
    expected_smiles = "not-a-smiles"
    assert canonicalize_mapped_smiles(input_smiles) == expected_smiles


def test_inchi_key():
    smiles = "c1c(Cl)cccc1"
    key = inchi_key(smiles)
    assert key == "MVPPADPHJFYWMZ-UHFFFAOYSA-N"

    smiles = "not-a-smiles"
    key = inchi_key(smiles)
    assert key == smiles


def test_uniqueify_smiles():
    smiles = [["CCO", "CO", "OCC"], ["CCSO", "OSCC", "SOC"], ["CO", "CCO", "CSO"]]
    log_lhs = [[-4, -6, -8], [-3.5, -4.2, -4.3], [-3.5, -4.2, -4.3]]
    smiles_unique, log_lhs_unique = uniqueify_sampled_smiles(smiles, log_lhs)

    smiles_expected = [["CCO", "CO"], ["CCSO", "COS"], ["CO", "CCO", "CSO"]]
    llhs_expected = [[-4, -6], [-3.5, -4.3], [-3.5, -4.2, -4.3]]

    assert len(smiles_unique) == len(smiles_expected)
    assert all(
        [
            output == expected
            for top_k, top_k_exp in zip(smiles_unique, smiles_expected)
            for output, expected in zip(top_k, top_k_exp)
        ]
    )

    assert all(
        [
            output == expected
            for top_k, top_k_exp in zip(log_lhs_unique, llhs_expected)
            for output, expected in zip(top_k, top_k_exp)
        ]
    )


def test_enumerate_smiles():
    smiles_list = ["CO", "CCO", "CSO"]
    enumerated_smiles = enumerate_smiles_combinations(smiles_list, 2, 3)
    expected_smiles = [["CO", "CCO"], ["CO", "CSO"], ["CCO", "CSO"], ["CO", "CCO", "CSO"]]

    assert len(enumerated_smiles) == len(expected_smiles)
    assert all(
        [
            output == expected
            for combination, combination_exp in zip(enumerated_smiles, expected_smiles)
            for output, expected in zip(combination, combination_exp)
        ]
    )


def test_smiles_to_fp():
    smiles = "c1ccccc1"
    fp1 = smiles_to_fingerprint(smiles, 2, 1024)
    assert fp1.sum() == 3
