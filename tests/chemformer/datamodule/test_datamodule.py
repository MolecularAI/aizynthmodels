import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from rdkit import Chem

from aizynthmodels.chemformer.data import (
    ChemblDataModule,
    ClassificationDataModule,
    SynthesisDataModule,
    ZincDataModule,
)


@pytest.fixture
def molecule_data():
    data = pd.DataFrame(
        {
            "molecules": [
                "O",
                "CC(=O)O",
                "CC(=O)C",
                "c1ccccc1",
                "Cc1ccccc1",
                "Oc1ccccc1",
                "C1CCOOC1",
                "CC(C)(C)O",
                "CC(C)(Cl)O",
                "CCN",
            ],
            "lengths": range(10),
            "set": [
                "train",
                "train",
                "train",
                "train",
                "test",
                "test",
                "test",
                "val",
                "val",
                "val",
            ],
        }
    )
    return data


@pytest.fixture
def reactants():
    return [
        "CC(C)(C)OC(=O)N1CC[C@H](NCc2ccccc2)[C@H](F)C1",
        "Nc1ncc(Br)nc1Br.C1COCCN1",
        "COC(=O)c1ccsc1NC(=O)NC(=O)C(Cl)(Cl)Cl.BrBr",
        "O=S(=O)(Cl)c1ccccc1.C1CNCCN1",
        "BrCc1ccccc1.O=Cc1cc(Br)ccc1O",
        "C[Si](C)(C)CCOCCl.Brc1ccc2[nH]ccc2c1",
    ]


@pytest.fixture
def products():
    return [
        "CC(C)(C)OC(=O)N1CC[C@H](N)[C@H](F)C1",
        "Nc1ncc(Br)nc1N1CCOCC1",
        "COC(=O)c1cc(Br)sc1NC(=O)NC(=O)C(Cl)(Cl)Cl",
        "O=S(=O)(c1ccccc1)N1CCNCC1",
        "O=Cc1cc(Br)ccc1OCc1ccccc1",
        "C[Si](C)(C)CCOCn1ccc2cc(Br)ccc21",
    ]


@pytest.fixture
def chembl_data_pkl(molecule_data, tmpdir):
    filename = str(tmpdir / "molecule_temp.pkl")
    molecule_data.to_pickle(filename)
    return filename


@pytest.fixture
def zinc_data_csv(molecule_data, tmpdir):
    filename = str(tmpdir / "molecule_temp.csv")
    molecule_data = molecule_data.rename(columns={"molecules": "smiles"})
    molecule_data.to_csv(filename)
    return filename


@pytest.fixture
def create_synthesis_data_file(reactants, products, tmpdir):
    filename = str(tmpdir / "synthesis_data_tmp.csv")
    data = pd.DataFrame(
        {
            "products": products,
            "reactants": reactants,
            "set": ["train", "test", "val", "train", "train", "test"],
        }
    )
    data.to_csv(filename, sep="\t", index=False)
    return filename


@pytest.fixture
def classification_data_file(products, tmpdir):
    filename = str(tmpdir / "synthesis_data_tmp.csv")
    data = pd.DataFrame(
        {
            "smiles": products,
            "label": [0, 2, 1, 1, 2, 0],
            "set": ["train", "test", "val", "train", "train", "test"],
        }
    )
    data.to_csv(filename, sep="\t", index=False)
    return filename


def test_chembl_datamodule(chembl_data_pkl, setup_masker):
    tokenizer, masker = setup_masker()
    datamodule = ChemblDataModule(
        dataset_path=chembl_data_pkl,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
    )
    datamodule.setup()

    assert datamodule.train_dataset[0] == {"molecules": "O", "lengths": 0}
    assert datamodule.test_dataset[0] == {"molecules": "Cc1ccccc1", "lengths": 4}
    assert datamodule.val_dataset[0] == {"molecules": "CC(C)(C)O", "lengths": 7}


def test_chembl_datamodule_collate_fn(chembl_data_pkl, setup_masker):
    tokenizer, masker = setup_masker()
    datamodule = ChemblDataModule(
        dataset_path=chembl_data_pkl,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
    )
    datamodule.setup()

    batch_data = [
        {
            "molecules": Chem.MolFromSmiles("O"),
        },
        {
            "molecules": Chem.MolFromSmiles("CC(=O)O"),
        },
        {
            "molecules": Chem.MolFromSmiles("CC(=O)C"),
        },
        {
            "molecules": Chem.MolFromSmiles("c1ccccc1"),
        },
    ]
    collate = datamodule.train_dataloader().collate_fn(batch_data)

    assert collate["target_smiles"] == ["O", "CC(=O)O", "CC(C)=O", "c1ccccc1"]


def test_zinc_datamodule(zinc_data_csv, setup_masker):
    tokenizer, masker = setup_masker()
    datamodule = ZincDataModule(
        dataset_path=zinc_data_csv,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
    )
    datamodule.setup()

    assert datamodule.train_dataset[0] == {"smiles": "O"}
    assert datamodule.test_dataset[0] == {"smiles": "Cc1ccccc1"}
    assert datamodule.val_dataset[0] == {"smiles": "CC(C)(C)O"}


def test_zinc_datamodule_collate_fn(zinc_data_csv, setup_masker):
    tokenizer, masker = setup_masker()
    datamodule = ZincDataModule(
        dataset_path=zinc_data_csv,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
    )
    datamodule.setup()

    batch_data = [
        {"smiles": "Cl"},
        {"smiles": "CC(=O)C"},
        {"smiles": "CC(=O)O"},
        {"smiles": "Brc1ccccc1"},
        {"smiles": "CC(C)(Cl)O"},
        {"smiles": "CCO"},
    ]
    collate = datamodule.train_dataloader().collate_fn(batch_data)

    assert collate["target_smiles"] == [
        "Cl",
        "CC(C)=O",
        "CC(=O)O",
        "Brc1ccccc1",
        "CC(C)(O)Cl",
        "CCO",
    ]


def test_synthesis_datamodule(create_synthesis_data_file, setup_tokenizer):
    datamodule = SynthesisDataModule(
        dataset_path=create_synthesis_data_file,
        tokenizer=setup_tokenizer(),
        batch_size=1,
        max_seq_len=100,
    )
    datamodule.setup()

    assert len(datamodule.train_dataloader()) == 3
    assert len(datamodule.test_dataloader()) == 2
    assert len(datamodule.val_dataloader()) == 1
    assert len(datamodule.full_dataloader()) == 6


def test_synthesis_datamodule_collate_fn(reactants, products, create_synthesis_data_file, setup_tokenizer):
    datamodule = SynthesisDataModule(
        dataset_path=create_synthesis_data_file,
        tokenizer=setup_tokenizer(),
        batch_size=1,
        max_seq_len=100,
        reactants=reactants,
        products=products,
        augmentation_strategy="all",
    )
    datamodule.setup()

    batch_data = [
        {
            "reactants": "CC(C)(C)OC(=O)N1CC[C@H](NCc2ccccc2)[C@H](F)C1",
            "products": "CC(C)(C)OC(=O)N1CC[C@H](N)[C@H](F)C1",
        },
        {
            "reactants": "O=S(=O)(Cl)c1ccccc1.C1CNCCN1",
            "products": "O=S(=O)(c1ccccc1)N1CCNCC1",
        },
        {
            "reactants": "BrCc1ccccc1.O=Cc1cc(Br)ccc1O",
            "products": "O=Cc1cc(Br)ccc1OCc1ccccc1",
        },
    ]
    collate = datamodule.train_dataloader().collate_fn(batch_data)

    assert collate["target_smiles"] == [
        "CC(C)(C)OC(=O)N1CC[C@H](N)[C@H](F)C1",
        "O=S(=O)(c1ccccc1)N1CCNCC1",
        "O=Cc1cc(Br)ccc1OCc1ccccc1",
    ]


def test_classification_datamodule(classification_data_file, setup_tokenizer):
    datamodule = ClassificationDataModule(
        dataset_path=classification_data_file,
        tokenizer=setup_tokenizer,
        batch_size=2,
        max_seq_len=100,
    )
    datamodule.setup()

    train_first_sample = datamodule.train_dataset[0]
    val_first_sample = datamodule.val_dataset[0]
    test_first_sample = datamodule.test_dataset[0]
    assert train_first_sample["input_smiles"] == "CC(C)(C)OC(=O)N1CC[C@H](N)[C@H](F)C1"
    assert train_first_sample["label"] == 0
    assert torch.equal(train_first_sample["class_indicator"], torch.tensor([1, 0, 0], dtype=torch.float64))

    assert val_first_sample["input_smiles"] == "COC(=O)c1cc(Br)sc1NC(=O)NC(=O)C(Cl)(Cl)Cl"
    assert val_first_sample["label"] == 1
    assert torch.equal(val_first_sample["class_indicator"], torch.tensor([0, 1, 0], dtype=torch.float64))

    assert test_first_sample["input_smiles"] == "Nc1ncc(Br)nc1N1CCOCC1"
    assert test_first_sample["label"] == 2
    assert torch.equal(test_first_sample["class_indicator"], torch.tensor([0, 0, 1], dtype=torch.float64))


def test_classifier_datamodule_collate_fn(products, setup_tokenizer):
    pl.seed_everything(1)
    datamodule = ClassificationDataModule(
        dataset_path="",
        smiles=products,
        labels=[0, 2, 1, 1, 2, 0],
        tokenizer=setup_tokenizer(),
        batch_size=2,
        max_seq_len=100,
        augment_prob=0.1,
    )
    datamodule.setup()

    batch_data = [
        {
            "input_smiles": "CC(C)(C)OC(=O)N1CC[C@H](NCc2ccccc2)[C@H](F)C1",
            "label": 0,
            "class_indicator": torch.tensor([1, 0, 0]),
        },
        {"input_smiles": "O=S(=O)(Cl)c1ccccc1.C1CNCCN1", "label": 2, "class_indicator": torch.tensor([0, 0, 1])},
        {"input_smiles": "BrCc1ccccc1.O=Cc1cc(Br)ccc1O", "label": 1, "class_indicator": torch.tensor([0, 1, 0])},
    ]
    collate = datamodule.train_dataloader().collate_fn(batch_data)
    assert torch.equal(collate["class_indicator"], torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
