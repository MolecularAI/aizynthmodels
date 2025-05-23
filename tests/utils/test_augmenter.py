import pytest
import pytorch_lightning as pl
from rdkit import Chem

from aizynthmodels.utils.tokenizer import MolAugmenter, SMILESAugmenter


@pytest.fixture
def smiles(shared_datadir):
    with open(shared_datadir / "test_smiles.smi") as file:
        smiles_data = file.readlines()
        test_smiles = [smi[:-1] for smi in smiles_data]
    return test_smiles


def get_num_new_random_smiles(smiles, test_smiles):
    num_new = 0
    for smi, smi_rand in zip(smiles, test_smiles):
        if smi != smi_rand:
            num_new += 1

    return num_new


def test_single_smiles_augment(smiles):
    """Checks the `SMLIESRandomizer` by testing that when `restricted` is
    `False` mostly (99%) of the SMILES randomized are distinct from the
    canonical.
    """
    smiles_randomizer = SMILESAugmenter(restricted=False)
    randomized_smiles = smiles_randomizer(smiles[0])
    assert len(randomized_smiles) == 1
    assert isinstance(randomized_smiles, list)


def test_failed_augmentation():
    smiles_randomizer = SMILESAugmenter(restricted=False)
    randomized_smiles = smiles_randomizer("not-a-smiles")
    assert randomized_smiles[0] == "not-a-smiles"

    smiles_randomizer = SMILESAugmenter(restricted=True)
    randomized_smiles = smiles_randomizer("not-a-smiles")
    assert randomized_smiles[0] == "not-a-smiles"


def test_smiles_majority_random_unrestricted(smiles):
    """Checks the `SMLIESRandomizer` by testing that when `restricted` is
    `False` mostly (99%) of the SMILES randomized are distinct from the
    canonical.
    """
    smiles_randomizer_unrestricted = SMILESAugmenter(restricted=False)
    randomized_smiles = smiles_randomizer_unrestricted(smiles)

    num_new = get_num_new_random_smiles(smiles, randomized_smiles)

    assert num_new / len(smiles) >= 0.9


def test_smiles_majority_random_restricted(smiles):
    """Checks the `SMLIESRandomizer` by testing that when `restricted` is
    `True` mostly (99%) of the SMILES randomized are distinct from the
    canonical.
    """
    smiles_randomizer_restricted = SMILESAugmenter(restricted=True)
    randomized_smiles = smiles_randomizer_restricted(smiles)

    num_new = get_num_new_random_smiles(smiles, randomized_smiles)

    assert num_new / len(smiles) >= 0.9


def test_mol_majority_random(smiles):
    """Checks the `MolRandomizer` by testing that mostly (99%) of the Mols
    randomized are distinct from the original canonical.
    """
    mol_randomizer = MolAugmenter()

    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    assert all([bool(mol) for mol in mols])

    mols_randomized = mol_randomizer(mols)

    randomized_smiles = [Chem.MolToSmiles(mol, canonical=False) for mol in mols_randomized]

    num_new = get_num_new_random_smiles(smiles, randomized_smiles)

    assert num_new / len(mols) >= 0.9


def test_mol_equality_random(smiles):
    """Check molecular equivalence after randomization by canonicalizing"""
    smiles_randomizer_unrestricted = SMILESAugmenter(restricted=False)
    randomized_smiles = smiles_randomizer_unrestricted(smiles)

    assert all(
        [
            Chem.MolToSmiles(Chem.MolFromSmiles(mol1)) == Chem.MolToSmiles(Chem.MolFromSmiles(mol2))
            for mol1, mol2 in zip(randomized_smiles, smiles)
        ]
    )


def test_mol_equality_restricted(smiles):
    """Check molecular equivalence after randomization by canonicalizing"""
    smiles_randomizer_unrestricted = SMILESAugmenter(restricted=True)
    randomized_smiles = smiles_randomizer_unrestricted(smiles)

    assert all(
        [
            Chem.MolToSmiles(Chem.MolFromSmiles(mol1)) == Chem.MolToSmiles(Chem.MolFromSmiles(mol2))
            for mol1, mol2 in zip(randomized_smiles, smiles)
        ]
    )


def test_active(smiles):
    """Tests that the `active` property works, i.e, that when the augmenter is
    not active it just returns the object that is input.
    """
    randomizer = SMILESAugmenter()
    smiles_rand = randomizer(smiles)

    assert smiles_rand != smiles

    randomizer.active = False
    smiles_nonrand = randomizer(smiles)

    assert smiles_nonrand == smiles


def test_mol_low_aug_prob(smiles):
    """Check that by setting a very low augment probability few new SMILES are generated"""
    pl.seed_everything(1)
    smiles_randomizer_unrestricted = SMILESAugmenter(restricted=False, augment_prob=0.1)
    randomized_smiles = smiles_randomizer_unrestricted(smiles)

    num_new = get_num_new_random_smiles(smiles, randomized_smiles)
    assert num_new / len(randomized_smiles) <= 0.2
    assert num_new >= 1


def test_mol_zero_aug_prob(smiles):
    """Check that by setting augment probability to zero, no new SMILES are generated"""
    smiles_randomizer_unrestricted = SMILESAugmenter(restricted=False, augment_prob=0.0)
    randomized_smiles = smiles_randomizer_unrestricted(smiles)

    num_new = get_num_new_random_smiles(smiles, randomized_smiles)
    assert num_new == 0


def test_mol_zero_aug_prob_restricted(smiles):
    """Check that by setting augment probability to zero, no new SMILES are generated"""
    smiles_randomizer_unrestricted = SMILESAugmenter(restricted=True, augment_prob=0.0)
    randomized_smiles = smiles_randomizer_unrestricted(smiles)

    num_new = get_num_new_random_smiles(smiles, randomized_smiles)
    assert num_new == 0
