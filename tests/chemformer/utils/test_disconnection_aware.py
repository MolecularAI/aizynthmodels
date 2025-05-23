import pytest
from rdkit import Chem

from aizynthmodels.chemformer.utils.disconnection_aware import (
    RXN_MAPPER,
    mapping_to_index,
    predictions_atom_mapping,
    propagate_input_mapping_to_reactants,
    tag_current_bond,
    verify_disconnection,
)


@pytest.mark.parametrize(
    ("reactants_smiles", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            {2: 0, 1: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6},
        ),
        (
            "[Cl:6].[CH:1]1=[CH:17][CH:2]=[CH:5][CH:24]=[CH:3]1",
            {6: 0, 1: 1, 17: 2, 2: 3, 5: 4, 24: 5, 3: 6},
        ),
    ],
)
def test_mapping_to_index(reactants_smiles, expected):
    mapping2idx = mapping_to_index(Chem.MolFromSmiles(reactants_smiles))
    assert mapping2idx == expected


@pytest.mark.parametrize(
    ("reactants", "product_new_mapping", "product_old_mapping", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:2][CH:7]=[CH:16]1",
            "[Cl:5].[cH:1]1[cH:2][cH:7][cH:16][cH:3][cH:15]1",
        ),
        (
            "[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:7][CH:8]=[CH:16]1",
            "[cH:1]1[cH:7][cH:8][cH:16][cH:3][cH:15]1",
        ),
    ],
)
def test_input_mapping_to_reactants(reactants, product_new_mapping, product_old_mapping, expected):
    assert propagate_input_mapping_to_reactants(product_old_mapping, reactants, product_new_mapping) == expected


@pytest.mark.skipif(RXN_MAPPER is None, reason="RXN-mapper not in env")
def test_predictions_atom_mapping():
    mapped_rxns, confidences = predictions_atom_mapping(["Clc1ccccc1"], [["c1ccccc1.Cl", "c1ccccc1.Br"]])

    assert len(mapped_rxns) == 2
    assert (
        mapped_rxns[0] == "[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[ClH:1]>>[Cl:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    )
    assert mapped_rxns[1] == "[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.Br>>[Cl:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    assert round(confidences[0], 4) == 0.4412
    assert round(confidences[1], 4) == 0.5961


@pytest.mark.skipif(RXN_MAPPER is not None, reason="RXN-mapper is in env")
def test_predictions_atom_mapping_import_error():
    with pytest.raises(ImportError, match="rxnmapper has to be installed in the environment"):
        predictions_atom_mapping(["Clc1ccccc1"], [["c1ccccc1.Cl", "c1ccccc1.Br"]])


@pytest.mark.parametrize(
    ("product_mapping", "bond_atom_inds", "expected"),
    [
        (
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:2][CH:6]=[CH:16]1",
            [1, 15],
            "Clc1ccc[cH:1][cH:1]1",
        ),
        (
            "[Cl:5][C:3]1=[CH:15][CH:1]=[CH:2][CH:6]=[CH:16]1",
            [5, 3],
            "c1cc[c:1]([Cl:1])cc1",
        ),
    ],
)
def test_tag_current_bond(product_mapping, bond_atom_inds, expected):
    assert tag_current_bond(product_mapping, bond_atom_inds) == expected


@pytest.mark.parametrize(
    ("reactants_smiles", "bond_atom_inds", "mapping_to_index", "expected"),
    [
        ("[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1", [1, 2], None, True),
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            [1, 2],
            {2: 0, 1: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6},
            True,
        ),
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            [6, 7],
            None,
            False,
        ),
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            [6, 7],
            {2: 0, 1: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6},
            False,
        ),
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            [8, 9],
            {2: 0, 1: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6},
            False,
        ),
    ],
)
def test_verify_disconnection(reactants_smiles, bond_atom_inds, mapping_to_index, expected):
    assert verify_disconnection(reactants_smiles, bond_atom_inds, mapping_to_index) == expected
