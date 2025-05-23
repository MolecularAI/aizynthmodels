from aizynthmodels.template_based.data import (
    InMemoryLabeledDataset,
    InMemoryLabeledReactionDataset,
    PrecomputedDataModule,
    SmilesBasedDataModule,
)


def test_create_dataset(dummy_data, reaction_data):
    dataset = InMemoryLabeledDataset(*dummy_data)
    assert len(dataset) == 10

    reactants, products, _ = reaction_data
    dataset = InMemoryLabeledReactionDataset(*dummy_data, reactants, products)
    assert len(dataset) == 10
    assert dataset.reactants == reaction_data[0]


def test_dataset_index(dummy_data, reaction_data):
    dataset = InMemoryLabeledDataset(*dummy_data)

    item = dataset[0]

    assert "input" in item
    assert "label" in item
    assert list(item["input"]) == [0, 1, 0]
    assert list(item["label"]) == [0, 1]

    reactants, products, _ = reaction_data
    dataset = InMemoryLabeledReactionDataset(*dummy_data, reactants, products)

    item = dataset[0]

    assert "input" in item
    assert "label" in item
    assert "reactant" in item
    assert "product" in item
    assert list(item["input"]) == [0, 1, 0]
    assert list(item["label"]) == [0, 1]
    assert item["reactant"] == "CCS(=O)(=O)Cl.OCCBr"
    assert item["product"] == "CCS(=O)(=O)OCCBr"


def test_create_precomputed_datamodule(dummy_datafiles):
    prefix = dummy_datafiles

    dm = PrecomputedDataModule(
        prefix,
        inputs_postfix="inputs.npz",
        labels_postfix="labels.npz",
        reactions_postfix=None,
        batch_size=2,
    )
    dm.setup()

    assert len(dm.test_dataset) == 1
    assert len(dm.val_dataset) == 1
    assert len(dm.train_dataset) == 8


def test_create_precomputed_datamodule_reaction_data(dummy_datafiles):
    prefix = dummy_datafiles

    dm = PrecomputedDataModule(
        prefix,
        inputs_postfix="inputs.npz",
        labels_postfix="labels.npz",
        reactions_postfix="reactions.csv",
        batch_size=2,
    )
    dm.setup()

    assert len(dm.test_dataset) == 1
    assert len(dm.val_dataset) == 1
    assert len(dm.train_dataset) == 8


def test_create_smiles_based_datamodule(dummy_datafiles):
    dataset_path = f"{dummy_datafiles}_reactions.csv"

    dm = SmilesBasedDataModule(
        dataset_path,
        fingerprint_size=256,
        fingerprint_radius=1,
        chirality=False,
        batch_size=2,
    )
    dm.setup()

    assert len(dm.test_dataset) == 1
    assert len(dm.val_dataset) == 1
    assert len(dm.train_dataset) == 8

    train_dataloader = dm.train_dataloader()
    batches = [batch for batch in train_dataloader]
    assert len(batches) == 4
    assert batches[0]["input"].numpy().shape == (2, 256)
    assert batches[0]["label"].numpy().shape == (2, 0)
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1
    assert len(dm.full_dataloader()) == 5


def test_train_dataloader(dummy_datafiles):
    prefix = dummy_datafiles
    dm = PrecomputedDataModule(
        prefix,
        inputs_postfix="inputs.npz",
        labels_postfix="labels.npz",
        reactions_postfix=None,
        batch_size=2,
        shuffle=False,
    )
    dm.setup()

    dataloader = dm.train_dataloader()

    assert len(dataloader) == 4

    batches = [batch for batch in dataloader]
    assert len(batches) == 4

    assert batches[0]["input"].numpy().shape == (2, 3)
    assert batches[0]["label"].numpy().shape == (2, 2)


def test_testval_dataloader(dummy_datafiles):
    prefix = dummy_datafiles
    dm = PrecomputedDataModule(
        prefix,
        inputs_postfix="inputs.npz",
        labels_postfix="labels.npz",
        reactions_postfix=None,
        batch_size=2,
        shuffle=False,
    )
    dm.setup()

    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1
