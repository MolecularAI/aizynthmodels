import pytest

from aizynthmodels.chemformer.data.base import ChemistryDataset, MoleculeListDataModule, ReactionListDataModule


@pytest.fixture
def create_smiles_file(tmpdir):
    filename = str(tmpdir / "smiles_temp.txt")

    def wrapper():
        with open(filename, "w") as fileobj:
            fileobj.write(
                "\n".join(
                    [
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
                    ]
                )
            )
        return filename

    return wrapper


@pytest.fixture
def create_reactions_file(tmpdir):
    filename = str(tmpdir / "rxns_temp.txt")

    def wrapper():
        with open(filename, "w") as fileobj:
            fileobj.write(
                "\n".join(
                    [
                        "O>>Cl",
                        "CC(=O)O>>CC(=O)C",
                        "CC(=O)C>>CC(=O)O",
                        "c1ccccc1>>c1ccccc1",
                        "Cc1ccccc1>>Brc1ccccc1",
                        "Oc1ccccc1>>Brc1ccccc1",
                        "C1CCOOC1>>C1CCOOC1",
                        "CC(C)(C)O>>CC(C)(C)O",
                        "CC(C)(Cl)O>>CC(C)(Cl)O",
                        "CCN>>CCO",
                    ]
                )
            )
        return filename

    return wrapper


def test_chemistry_dataset():
    data = ChemistryDataset({"lengths": [1, 2, 3], "b": [True, False, True]})

    assert len(data) == 3
    assert data[1] == {"lengths": 2, "b": False}
    assert data.seq_lengths == [1, 2, 3]


def test_chemistry_dataset_empty_data():
    data = ChemistryDataset({})

    assert len(data) == 0


def test_chemistry_dataset_no_seq_lengths():
    data = ChemistryDataset({"a": [1, 2, 3], "b": [True, False, True]})

    with pytest.raises(KeyError, match="does not store any sequence lengths"):
        _ = data.seq_lengths


def test_molecule_list_datamodule(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker()
    datamodule = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
    )
    datamodule.setup()

    assert len(datamodule.train_dataloader()) == 3
    assert len(datamodule.test_dataloader()) == 1
    assert len(datamodule.val_dataloader()) == 1
    assert len(datamodule.full_dataloader()) == 5


def test_molecule_list_datamodule_test_idxs(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker()
    datamodule = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
        test_idxs=[0, 1, 2, 3],
    )
    datamodule.setup()

    # Random sampler for training cannot handle empty sets
    assert len(datamodule.train_dataloader()) == 3
    assert len(datamodule.test_dataloader()) == 2
    assert len(datamodule.val_dataloader()) == 0
    assert len(datamodule.full_dataloader()) == 5


def test_molecule_list_datamodule_val_idxs(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker()
    datamodule = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
        val_idxs=[0, 1, 2, 3],
    )
    datamodule.setup()

    assert len(datamodule.train_dataloader()) == 3
    assert len(datamodule.test_dataloader()) == 0
    assert len(datamodule.val_dataloader()) == 2
    assert len(datamodule.full_dataloader()) == 5


def test_molecule_list_datamodule_test_val_idxs(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker()
    datamodule = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
        test_idxs=[4, 5, 6, 7, 8, 9],
        val_idxs=[0, 1, 2, 3],
    )
    datamodule.setup()

    with pytest.raises(ValueError):
        datamodule.train_dataloader()

    assert len(datamodule.test_dataloader()) == 3
    assert len(datamodule.val_dataloader()) == 2
    assert len(datamodule.full_dataloader()) == 5


def test_molecule_list_datamodule_nchunks_greater_than_one(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker()
    datamodule = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
        n_chunks=2,
    )
    datamodule.setup()

    assert len(datamodule.train_dataloader()) == 1
    assert len(datamodule.test_dataloader()) == 0
    assert len(datamodule.val_dataloader()) == 0
    assert len(datamodule.full_dataloader()) == 2


def test_molecule_list_datamodule_masker_error(create_smiles_file, setup_tokenizer):
    dataset_path = create_smiles_file()

    with pytest.raises(ValueError, match="Need to provide a masker with task"):
        MoleculeListDataModule(
            dataset_path=dataset_path,
            tokenizer=setup_tokenizer(),
            batch_size=2,
            max_seq_len=100,
            n_chunks=2,
        )


def test_molecule_list_datamodule_collate_fn(create_smiles_file, setup_masker):
    dataset_path = create_smiles_file()
    tokenizer, masker = setup_masker()
    datamodule = MoleculeListDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        masker=masker,
        batch_size=2,
        max_seq_len=100,
        n_chunks=2,
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


def test_reaction_list_datamodule_same_reactants_products(create_smiles_file, setup_tokenizer):
    dataset_path = create_smiles_file()
    datamodule = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=10,
        max_seq_len=100,
    )
    datamodule.setup()

    assert datamodule.val_dataset[0]["reactants"] == datamodule.val_dataset[0]["products"]


def test_reaction_list_datamodule(create_reactions_file, setup_tokenizer):
    dataset_path = create_reactions_file()
    datamodule = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=10,
        max_seq_len=100,
    )
    datamodule.setup()
    datamodule_reverse = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=10,
        max_seq_len=100,
        reverse=True,
    )
    datamodule_reverse.setup()

    batch = next(iter(datamodule.full_dataloader()))
    batch_reverse = next(iter(datamodule_reverse.full_dataloader()))

    assert batch["encoder_input"][1:, :].tolist() != batch_reverse["decoder_input"].tolist()
    assert batch["decoder_input"].tolist() != batch_reverse["encoder_input"][1:, :].tolist()


def test_reaction_list_datamodule_collate_fn(create_reactions_file, setup_tokenizer):
    dataset_path = create_reactions_file()
    datamodule = ReactionListDataModule(
        dataset_path=dataset_path,
        tokenizer=setup_tokenizer(),
        batch_size=2,
        max_seq_len=100,
        n_chunks=2,
    )
    datamodule.setup()

    batch_data = [
        {"reactants": "O", "products": "Cl"},
        {"reactants": "CC(=O)O", "products": "CC(=O)C"},
        {"reactants": "CC(=O)C", "products": "CC(=O)O"},
        {"reactants": "Oc1ccccc1", "products": "Brc1ccccc1"},
        {"reactants": "CC(C)(Cl)O", "products": "CC(C)(Cl)O"},
        {"reactants": "CCN", "products": "CCO"},
    ]
    collate = datamodule.train_dataloader().collate_fn(batch_data)

    assert collate["target_smiles"] == [
        "Cl",
        "CC(=O)C",
        "CC(=O)O",
        "Brc1ccccc1",
        "CC(C)(Cl)O",
        "CCO",
    ]
