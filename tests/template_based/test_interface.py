import pytest
import pytorch_lightning as pl
from omegaconf import OmegaConf

from aizynthmodels.template_based.data import PrecomputedDataModule
from aizynthmodels.template_based.data import __name__ as data_module
from aizynthmodels.template_based.models.interface import TemplateBasedRetrosynthesis
from aizynthmodels.utils.loading import build_datamodule


@pytest.fixture
def default_config(model_hyperparams, dummy_datafiles):
    datamodule_config = {
        "type": "PrecomputedDataModule",
        "arguments": [
            {"files_prefix": dummy_datafiles},
            {"inputs_postfix": "inputs.npz"},
            {"labels_postfix": "labels.npz"},
            {"reactions_postfix": None},
        ],
    }

    config = OmegaConf.create(
        {
            "file_prefix": dummy_datafiles,
            "mode": "train",
            "n_predictions": 2,
            "device": "cpu",
            "random_seed": 1,
            "batch_size": 2,
            "dataset_part": "test",
            "model_path": None,
            "datamodule": datamodule_config,
            "model_hyperparams": model_hyperparams,
        }
    )
    return config


def test_set_datamodule(default_config):

    model = TemplateBasedRetrosynthesis(default_config)

    model.set_datamodule(datamodule_config=default_config.datamodule)
    assert isinstance(model.datamodule, PrecomputedDataModule)

    model.datamodule = None
    model.set_datamodule()
    assert not model.datamodule

    datamodule = build_datamodule(default_config.datamodule, data_module)
    model.set_datamodule(datamodule=datamodule)
    assert isinstance(model.datamodule, PrecomputedDataModule)


def test_prediction_output(default_config):
    pl.seed_everything(1)
    model = TemplateBasedRetrosynthesis(default_config)
    predictions = model.predict()
    expected_probs = [0.59, 0.41]

    assert list(predictions.keys()) == ["predictions", "probabilities", "ground_truth"]
    assert len(predictions["predictions"]) == 1
    assert all(round(prob, 2) == exp_prob for prob, exp_prob in zip(predictions["probabilities"][0], expected_probs))
