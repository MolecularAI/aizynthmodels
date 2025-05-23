"""Module containing auxiliary functions needed to run the disconnection-Chemformer"""
import numpy as np
from rdkit import Chem
from rxnutils.chem.utils import remove_atom_mapping

RXN_MAPPER_ENV_OK = True
try:
    from rxnmapper import RXNMapper

    RXN_MAPPER = RXNMapper()
except ImportError:
    RXN_MAPPER = None

from typing import Dict, List, Optional, Tuple  # noqa: E402

from aizynthmodels.utils.smiles import canonicalize_mapped_smiles, construct_reaction_smiles  # noqa: E402


def mapping_to_index(mol: Chem.rdchem.Mol) -> Dict[int, int]:
    """
    Atom-map-num to index mapping.

    :param mol: rdkit Molecule
    :return: a dictionary which maps atom-map-number to atom-index"""
    mapping = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
    return mapping


def predictions_atom_mapping(
    smiles_list: List[str], predicted_smiles_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create atom-mapping on the predicted reactions using RXN-mapper.
    Requires RXNMapper to be installed.

    :param smiles_list: batch of input product SMILES to predict atom-mapping on
    :param predicted_smiles_list: batch of predicted reactant SMILES
    :return: the atom-mapped reactions and the mapping confidence
    """
    if not RXN_MAPPER:
        raise ImportError("rxnmapper has to be installed in the environment!")
    rxn_smiles_list = []
    for product_smiles_mapped, reactants_smiles in zip(smiles_list, predicted_smiles_list):
        product_smiles = remove_atom_mapping(product_smiles_mapped)

        rxn_smiles_list.extend(construct_reaction_smiles(product_smiles, reactants_smiles))

    mapped_rxns = RXN_MAPPER.get_attention_guided_atom_maps(rxn_smiles_list, canonicalize_rxns=False)

    atom_map_confidences = np.array([rxnmapper_output["confidence"] for rxnmapper_output in mapped_rxns])
    mapped_rxns = np.array([rxnmapper_output["mapped_rxn"] for rxnmapper_output in mapped_rxns])
    return mapped_rxns, atom_map_confidences


def propagate_input_mapping_to_reactants(
    product_input_mapping: str,
    predicted_reactants: str,
    product_new_mapping: str,
) -> str:
    """
    Propagate old atom-mapping to reactants using the new atom-mapping.

    :param product_input_mapping: input product.
    :param predicted_reactants: predicted_reactants without atom-mapping.
    :param product_new_mapping: product with new mapping from rxn-mapper.
    :return: reactant SMILES with the same atom-mapping as the input product.
    """

    product_input_mapping = canonicalize_mapped_smiles(product_input_mapping)
    product_new_mapping = canonicalize_mapped_smiles(product_new_mapping)

    mol_input_mapping = Chem.MolFromSmiles(product_input_mapping)
    mol_new_mapping = Chem.MolFromSmiles(product_new_mapping)

    reactants_mol = Chem.MolFromSmiles(predicted_reactants)
    reactants_map_to_index = mapping_to_index(reactants_mol)
    predicted_reactants = remove_atom_mapping(predicted_reactants, canonical=False)
    reactants_mol = Chem.MolFromSmiles(predicted_reactants)

    for atom_idx, atom_input in enumerate(mol_input_mapping.GetAtoms()):
        atom_new_mapping = mol_new_mapping.GetAtomWithIdx(atom_idx)

        atom_map_num_input = atom_input.GetAtomMapNum()
        atom_map_num_new_mapping = atom_new_mapping.GetAtomMapNum()

        try:
            atom_reactant = reactants_mol.GetAtomWithIdx(reactants_map_to_index[atom_map_num_new_mapping])
            atom_reactant.SetAtomMapNum(atom_map_num_input)
        except KeyError:
            continue

    return Chem.MolToSmiles(reactants_mol)


def tag_current_bond(product_smiles: str, bond_inds: List[int]) -> str:
    """
    Remove atom-tagging on all atoms except those in bonds_inds.
    Tag bond_inds atoms as [<atom>:1] where <atom> is any atom.

    :param product_smiles: SMILES with atom-mapping
    :param bond_inds: atom indices involved in current bond to break
    :return: atom-map tagged SMILES
    """
    mol = Chem.MolFromSmiles(product_smiles)

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in bond_inds:
            atom.SetAtomMapNum(1)
        else:
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def verify_disconnection(
    reactants_mapped: str,
    bond_atom_inds: List[int],
    mapping_to_index: Optional[Dict[int, int]] = None,
) -> bool:
    """
    Check that current bond was broken in a specific prediction (reactants_smiles).

    :param reactants_mapped: Atom-mapped reactant SMILES
    :param bond_atom_inds: The bond which should be broken
    :param mapping_to_index: Dictionary mapping atom-map number to atom idx in the
        reactants SMILES.
    :return: whether the bonds was broken or not.
    """
    mol = Chem.MolFromSmiles(reactants_mapped)

    if not mapping_to_index:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in bond_atom_inds:
                if any(neighbor.GetAtomMapNum() in bond_atom_inds for neighbor in atom.GetNeighbors()):
                    return False
    else:
        unmatched_count = 0
        for ind in bond_atom_inds:
            atom_idx = mapping_to_index.get(ind)
            if atom_idx is None:
                unmatched_count += 1
                continue
            atom = mol.GetAtomWithIdx(atom_idx)
            if any(neighbor.GetAtomMapNum() in bond_atom_inds for neighbor in atom.GetNeighbors()):
                return False
        if unmatched_count == len(bond_atom_inds):
            return False
    return True
