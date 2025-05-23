import json
import pickle

import omegaconf as oc
import pytest

from aizynthmodels.route_distances.data.datamodule import InMemoryTreeDataset
from aizynthmodels.route_distances.models import LstmDistanceModel
from aizynthmodels.route_distances.utils.data import collate_batch, collate_trees
from aizynthmodels.utils.trainer import instantiate_scorers


@pytest.fixture
def load_reaction_tree(shared_datadir):
    def wrapper(filename, index=0):
        filename = str(shared_datadir / filename)
        with open(filename, "r") as fileobj:
            trees = json.load(fileobj)
        if isinstance(trees, dict):
            return trees
        elif index == -1:
            return trees
        else:
            return trees[index]

    return wrapper


@pytest.fixture
def mock_distance_model(mocker):
    class MockedLstmDistanceModel(mocker.MagicMock):
        @property
        def hparams(self):
            pass

    class MockedHparams(mocker.MagicMock):
        @property
        def fingerprint_size(self):
            return 1024

    mocker.patch.object(MockedLstmDistanceModel, "hparams", MockedHparams())
    patched_model_cls = mocker.patch("aizynthmodels.route_distances.utils.calculator.LstmDistanceModel")
    patched_model_cls.load_from_checkpoint.return_value = MockedLstmDistanceModel()
    return patched_model_cls


def test_dummy_distance_model(shared_datadir, mocker):
    pickle_path = str(shared_datadir / "test_data.pickle")
    with open(pickle_path, "rb") as fileobj:
        data = pickle.load(fileobj)
    dataset = InMemoryTreeDataset(**data)
    batch = collate_batch([batch for batch in dataset])

    config = oc.OmegaConf.create(
        {"fp_size": 32, "lstm_size": 16, "dropout_prob": 0.0, "learning_rate": 0.01, "weight_decay": 0.01}
    )
    scores = instantiate_scorers(["R2Score", "MeanAbsoluteError"])

    model = LstmDistanceModel(config)
    model.scores = scores
    model.trainer = mocker.MagicMock()
    model._current_fx_name = "training_step"

    assert model.forward(collate_trees(data["trees"])).shape[0] == 45

    train_data = model.training_step(batch, None)
    assert round(train_data.tolist()) == pytest.approx(12, rel=1)

    val_data = model.validation_step(batch, None)
    assert all(key in val_data for key in ["val_loss", "val_mae", "val_r2"])
    assert val_data["val_mae"] == pytest.approx(1.93, rel=1e-2)
    model.on_validation_epoch_end()
    assert not model.validation_step_outputs

    test_data = model.test_step(batch, None)
    assert all(key in test_data for key in ["test_loss", "test_mae", "test_r2"])

    optimizer, scheduler = model.configure_optimizers()

    assert optimizer[0].state_dict()["param_groups"][0]["lr"] == config.learning_rate
    assert optimizer[0].state_dict()["param_groups"][0]["betas"] == (0.9, 0.999)
    assert scheduler[0]["scheduler"].eps == 1e-08
