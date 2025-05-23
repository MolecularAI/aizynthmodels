from itertools import combinations
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rxnutils.chem.utils import has_atom_mapping, remove_atom_mapping, split_smiles_from_reaction


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize smiles and sort the (possible) multiple molcules.

    Args:
        smiles: Input SMILES string.
    Returns:
        Canonicalized SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles

    smiles_canonical = Chem.MolToSmiles(mol)
    smiles_canonical = ".".join(sorted(split_smiles_from_reaction(smiles_canonical)))
    return smiles_canonical


def canonicalize_mapped_smiles(smiles_mapped: str) -> str:
    """
    Return the canonical form of an atom-mapped SMILES, determined by the
    SMILES without atom-mapping.
    """
    if not has_atom_mapping(smiles_mapped):
        return canonicalize_smiles(smiles_mapped)

    smiles = remove_atom_mapping(smiles_mapped, canonical=False)

    mol_mapped = Chem.MolFromSmiles(smiles_mapped)
    mol_unmapped = Chem.MolFromSmiles(smiles)

    _, canonical_atom_order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol_unmapped))])))

    mol_mapped = Chem.RenumberAtoms(mol_mapped, canonical_atom_order)
    return Chem.MolToSmiles(mol_mapped, canonical=False)


def construct_reaction_smiles(product_smiles: str, reactants_smiles: np.ndarray) -> List[str]:
    """
    Construct the reaction smiles given product and reactant SMILES.

    :param product_smiles: input product SMILES
    :param reactants_smiles: list of predicted reactant SMILES
    :return: list of reaction SMILES
    """
    rxn_smiles = [f"{reactants}>>{product_smiles}" for reactants in reactants_smiles]
    return rxn_smiles


def inchi_key(smiles: str):
    """
    Get inchi key of input SMILES.

    Args:
        smiles: Input SMILES string
    Returns:
        Inchi-key of SMILES string or SMILES string if invalid rdkit molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles
    return Chem.MolToInchiKey(mol)


def uniqueify_sampled_smiles(
    sampled_smiles: List[np.ndarray],
    log_lhs: List[np.ndarray],
    n_unique_beams: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get unique SMILES and corresponding highest log-likelihood of each input.
    For beam_size > 1: Uniqueifying sampled molecules and select
    'n_unique_beams'-top molecules.

    Args:
        sampled_smiles: list of top-k sampled SMILES
        log_lhs: list of top-k log-likelihoods
        n_unique_beams: upper limit on number of unique SMILES to return
    Returns:
        Tuple of lists with unique SMILES and their corresponding highest
        log-likelihoods.
    """
    if not n_unique_beams:
        n_unique_beams = max([len(top_smiles) for top_smiles in sampled_smiles])

    sampled_smiles_unique = []
    log_lhs_unique = []
    for top_k_smiles, top_k_llhs in zip(sampled_smiles, log_lhs):
        top_k_mols = [Chem.MolFromSmiles(smi) for smi in top_k_smiles]
        top_k_smiles = [Chem.MolToSmiles(mol) for mol in top_k_mols if mol]
        top_k_llhs = [llhs for llhs, mol in zip(top_k_llhs, top_k_mols) if mol]

        top_k_unique = pd.DataFrame(
            {
                "smiles": top_k_smiles,
                "log_likelihood": top_k_llhs,
            }
        )
        top_k_unique.drop_duplicates(subset=["smiles"], keep="first", inplace=True)

        sampled_smiles_unique.append(list(top_k_unique["smiles"].values[0:n_unique_beams]))
        log_lhs_unique.append(list(top_k_unique["log_likelihood"].values[0:n_unique_beams]))

    return (
        sampled_smiles_unique,
        log_lhs_unique,
    )


def enumerate_smiles_combinations(smiles_list: List[str], min_n_combinations, max_n_combinations) -> List[List[str]]:
    """
    Returns all possible combinations of the molecules in smiles_list.
    """
    smiles_combinations = []
    for n_components in range(min_n_combinations, max_n_combinations + 1):
        for this_combination in combinations(smiles_list, n_components):
            smiles_combinations.append(list(this_combination))
    return smiles_combinations


def seq_smiles_to_fingerprint(
    smiles: Sequence[str], fp_radius: int, fp_length: int, chirality: bool = False
) -> np.ndarray:
    """Wrapper around `smiles_to_fingerprint` for `np.apply`"""
    return smiles_to_fingerprint(smiles[0], fp_radius, fp_length, chirality)


def seq_rxn_smiles_to_fingerprint(
    smiles: Sequence[str], fp_radius: int, fp_length: int, chirality: bool = False
) -> np.ndarray:
    """Wrapper around `reaction_smiles_to_fingerprint` for `np.apply`"""
    product_smiles, reactants_smiles = smiles
    return reaction_smiles_to_fingerprint(reactants_smiles, product_smiles, fp_radius, fp_length, chirality)


def smiles_to_fingerprint(
    smiles: str, fp_radius: int, fp_length: int, chirality: bool = False, **kwargs: Any
) -> np.ndarray:
    """
    Compute fingerprint from SMILES.

    :param smiles: a SMILES
    :param fp_radius: fingerprint radius.
    :param fp_length: fingerprint length.
    :param chirality: whether to use chirality
    :return: the SMILES' fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)

    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_length, useChirality=chirality, **kwargs)
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def reaction_smiles_to_fingerprint(
    reactants_smiles: str, product_smiles: str, fp_radius: int, fp_length: int, chirality: bool = False, **kwargs: Any
) -> np.ndarray:
    """
    Compute difference fingerprint from reaction SMILES.

    :param smiles: a SMILES
    :param fp_radius: fingerprint radius.
    :param fp_length: fingerprint length.
    :param chirality: whether to use chirality
    :return: the reactions fingerprint
    """
    product_fp = smiles_to_fingerprint(product_smiles, fp_radius, fp_length, chirality, **kwargs)

    reactants_fp_list = []
    for smiles in split_smiles_from_reaction(reactants_smiles):
        reactants_fp_list.append(smiles_to_fingerprint(smiles, fp_radius, fp_length, chirality, **kwargs))

    return (product_fp - sum(reactants_fp_list)).astype(np.int8)
