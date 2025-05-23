import pytest
from omegaconf import OmegaConf

from aizynthmodels.quick_filter.data import PrecomputedDataModule
from aizynthmodels.quick_filter.data import __name__ as data_module
from aizynthmodels.quick_filter.models.interface import QuickFilter
from aizynthmodels.utils.loading import build_datamodule


@pytest.fixture
def default_config(model_hyperparams, dummy_datafiles):
    datamodule_config = {
        "type": "PrecomputedDataModule",
        "arguments": [
            {"files_prefix": dummy_datafiles},
            {"inputs_prod_postfix": "inputs_prod.npz"},
            {"inputs_rxn_postfix": "inputs_rxn.npz"},
            {"labels_postfix": "labels.npz"},
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
            "threshold": 0.5,
        }
    )
    return config


def test_set_datamodule(default_config):

    model = QuickFilter(default_config)

    model.set_datamodule(datamodule_config=default_config.datamodule)
    assert isinstance(model.datamodule, PrecomputedDataModule)

    model.datamodule = None
    model.set_datamodule()
    assert not model.datamodule

    datamodule = build_datamodule(default_config.datamodule, data_module)
    model.set_datamodule(datamodule=datamodule)
    assert isinstance(model.datamodule, PrecomputedDataModule)


def test_prediction_output(default_config):

    model = QuickFilter(default_config)
    predictions = model.predict()

    assert list(predictions.keys()) == ["predictions", "probabilities", "ground_truth"]
    assert len(predictions["predictions"]) == 1
    assert predictions["probabilities"][0] > 0.0
    assert predictions["predictions"][0] in [0.0, 1.0]
