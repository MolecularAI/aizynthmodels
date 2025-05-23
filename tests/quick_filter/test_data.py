from aizynthmodels.quick_filter.data import InMemoryLabeledDataset, PrecomputedDataModule, SmilesBasedDataModule


def test_create_dataset(dummy_data, reaction_data):
    inputs, labels = dummy_data
    dataset = InMemoryLabeledDataset(inputs, inputs, labels)
    assert len(dataset) == 10


def test_dataset_index(dummy_data, reaction_data):
    inputs, labels = dummy_data
    dataset = InMemoryLabeledDataset(inputs, inputs, labels)

    item = dataset[0]

    assert "product_input" in item
    assert "reaction_input" in item
    assert "label" in item
    assert list(item["product_input"]) == [0, 1, 0]
    assert list(item["reaction_input"]) == [0, 1, 0]
    assert item["label"] == 1.0


def test_create_precomputed_datamodule(dummy_datafiles):
    prefix = dummy_datafiles

    dm = PrecomputedDataModule(
        prefix,
        inputs_prod_postfix="inputs_prod.npz",
        inputs_rxn_postfix="inputs_rxn.npz",
        labels_postfix="labels.npz",
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
    assert batches[0]["product_input"].numpy().shape == (2, 256)
    assert batches[0]["reaction_input"].numpy().shape == (2, 256)
    assert batches[0]["label"].numpy().shape == (2,)
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1
    assert len(dm.full_dataloader()) == 5


def test_train_dataloader(dummy_datafiles):
    prefix = dummy_datafiles
    dm = PrecomputedDataModule(
        prefix,
        inputs_prod_postfix="inputs_prod.npz",
        inputs_rxn_postfix="inputs_rxn.npz",
        labels_postfix="labels.npz",
        batch_size=2,
        shuffle=False,
    )
    dm.setup()

    dataloader = dm.train_dataloader()

    assert len(dataloader) == 4

    batches = [batch for batch in dataloader]
    assert len(batches) == 4

    assert batches[0]["product_input"].numpy().shape == (2, 3)
    assert batches[0]["reaction_input"].numpy().shape == (2, 3)
    assert batches[0]["label"].numpy().shape == (2,)


def test_testval_dataloader(dummy_datafiles):
    prefix = dummy_datafiles
    dm = PrecomputedDataModule(
        prefix,
        inputs_prod_postfix="inputs_prod.npz",
        inputs_rxn_postfix="inputs_rxn.npz",
        labels_postfix="labels.npz",
        batch_size=2,
        shuffle=False,
    )
    dm.setup()

    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1
