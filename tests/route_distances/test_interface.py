import pickle
import random

import pytest
import torch
from omegaconf import OmegaConf

from aizynthmodels.route_distances.data.datamodule import InMemoryTreeDataset, TreeDataModule
from aizynthmodels.route_distances.interface import RouteDistanceModel
from aizynthmodels.route_distances.utils.data import collate_batch

random.seed(a=1)
torch.manual_seed(1)


@pytest.fixture
def config(shared_datadir):
    model_hyperparams = {
        "fp_size": 2048,
        "lstm_size": 1024,
        "dropout_prob": 0.5,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
    }
    config = OmegaConf.create(
        {
            "model_path": None,
            "model_hyperparams": model_hyperparams,
            "datamodule": None,
            "data_path": None,
            "mode": "eval",
            "device": None,
        }
    )
    return config


def test_set_datamodule(shared_datadir, config):
    config.data_path = str(shared_datadir / "test_data.pickle")

    model = RouteDistanceModel(config)

    datamodule_config = OmegaConf.create({"type": "TreeDataModule"})
    model.set_datamodule(datamodule_config=datamodule_config)
    assert isinstance(model.datamodule, TreeDataModule)

    model.datamodule = None
    model.set_datamodule()
    assert not model.datamodule

    model.datamodule = None
    tree_datamodule = TreeDataModule(
        dataset_path=str(shared_datadir / "test_data.pickle"),
        batch_size=2,
    )
    model.set_datamodule(datamodule=tree_datamodule)
    assert isinstance(model.datamodule, TreeDataModule)


def test_on_device(shared_datadir, config):
    pickle_path = str(shared_datadir / "test_data.pickle")
    with open(pickle_path, "rb") as fileobj:
        data = pickle.load(fileobj)
    dataset = InMemoryTreeDataset(**data)
    batch = collate_batch([batch for batch in dataset])

    config.device = "cpu"
    model = RouteDistanceModel(config)
    output = model.on_device(batch)

    assert list(output.keys()) == ["tree1", "tree2", "ted"]
    assert torch.round(output["ted"]).tolist() == [0.0, 4.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 0.0, 10.0]
