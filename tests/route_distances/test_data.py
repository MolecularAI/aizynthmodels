import json

import pytest
import torch

from aizynthmodels.route_distances.data.datamodule import InMemoryTreeDataset, TreeDataModule, TreeListDataModule


@pytest.fixture
def dummy_dataset_input():
    trees = ["tree1", "tree2", "tree3"]
    pairs = [(0, 1, 0.5), (0, 2, 0.7)]
    return pairs, trees


def test_create_dataset(dummy_dataset_input):
    dataset = InMemoryTreeDataset(*dummy_dataset_input)

    assert len(dataset) == 2


def test_dataset_indexing(dummy_dataset_input):
    dataset = InMemoryTreeDataset(*dummy_dataset_input)

    assert dataset[0] == {"tree1": "tree1", "tree2": "tree2", "ted": 0.5}

    assert dataset[1] == {
        "tree1": "tree1",
        "tree2": "tree3",
        "ted": 0.7,
    }

    with pytest.raises(IndexError):
        _ = dataset[2]


def test_setup_datamodule(shared_datadir):
    pickle_path = str(shared_datadir / "test_data.pickle")
    data = TreeDataModule(pickle_path, batch_size=2, split_part=0.2)

    data.setup()

    assert len(data.train_dataset) == 6
    assert len(data.val_dataset) == 2
    assert len(data.test_dataset) == 2

    assert len(data.train_dataloader()) == 3
    assert len(data.test_dataloader()) == 1
    assert len(data.val_dataloader()) == 1
    assert len(data.full_dataloader()) == 5

    assert [idx2 for _, idx2, _ in data.train_dataset.pairs] == [0, 1, 8, 9, 2, 3]
    assert [idx2 for _, idx2, _ in data.val_dataset.pairs] == [6, 7]
    assert [idx2 for _, idx2, _ in data.test_dataset.pairs] == [4, 5]


def test_train_dataloader(shared_datadir):
    pickle_path = str(shared_datadir / "test_data.pickle")
    data = TreeDataModule(pickle_path, batch_size=2, shuffle=False, split_part=0.2)
    data.setup()

    dataloader = data.train_dataloader()

    assert len(dataloader) == 3

    batches = [batch for batch in dataloader]
    assert len(batches) == 3

    # Do some checks on the structure, but not everything
    assert len(batches[0]["ted"]) == 2
    assert len(batches[0]["tree1"]["tree_sizes"]) == 2
    assert batches[0]["tree1"]["tree_sizes"][0] in [3, 5]
    assert batches[0]["tree1"]["tree_sizes"][1] in [3, 5]


def test_val_and_test_dataloader(shared_datadir):
    pickle_path = str(shared_datadir / "test_data.pickle")
    data = TreeDataModule(pickle_path, batch_size=2, split_part=0.2)
    data.setup()

    assert len(data.val_dataloader()) == 1
    assert len(data.test_dataloader()) == 1


def test_prediction_datamodule(shared_datadir):
    # Initializing .json data file
    data_path = str(shared_datadir / "example_routes.json")
    data = TreeListDataModule(data_path)
    data.setup(fp_size=2048)

    dataloader = data.full_dataloader()

    assert len(dataloader) == 1

    batches = [batch for batch in dataloader]
    assert len(batches) == 1

    with open(data_path, "r") as fid:
        route_list = json.load(fid)

    # Initializing with pre-loaded route list
    data_list = TreeListDataModule(route_list=route_list)
    data_list.setup(fp_size=2048)
    batches_list = [batch for batch in data_list.full_dataloader()]

    assert all(
        torch.equal(batch["tree"]["features"], batch_list["tree"]["features"])
        for batch, batch_list in zip(batches, batches_list)
    )
    assert all(
        torch.equal(batch["tree"]["node_order"], batch_list["tree"]["node_order"])
        for batch, batch_list in zip(batches, batches_list)
    )
    assert all(
        torch.equal(batch["tree"]["edge_order"], batch_list["tree"]["edge_order"])
        for batch, batch_list in zip(batches, batches_list)
    )
    assert all(
        torch.equal(batch["tree"]["adjacency_list"], batch_list["tree"]["adjacency_list"])
        for batch, batch_list in zip(batches, batches_list)
    )

    # Do some checks on the structure, but not everything
    assert len(batches[0]["tree"]["tree_sizes"]) == 3
    assert batches[0]["tree"]["tree_sizes"][0] in [3, 5, 5]
    assert len(batches_list[0]["tree"]["tree_sizes"]) == 3
    assert batches_list[0]["tree"]["tree_sizes"][0] in [3, 5, 5]
