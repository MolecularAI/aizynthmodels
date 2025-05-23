import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from aizynthmodels.quick_filter.pipelines.scripts.create_filter_lib import main as create_filter_lib
from aizynthmodels.quick_filter.pipelines.scripts.featurize import main as featurize
from aizynthmodels.quick_filter.pipelines.scripts.split_data import main as split_data
from aizynthmodels.utils.configs.quick_filter.pipelines import FilterModelPipeline
from aizynthmodels.utils.hydra import load_config


@pytest.fixture
def setup_config(shared_datadir, tmpdir):
    filename = tmpdir / "config.yaml"
    with open(filename, "w") as fileobj:
        yaml.dump(
            {
                "file_prefix": str(shared_datadir / "big_dummy"),
                "n_batches": 1,
                "training_fraction": 0.6,
                "output_directory": str(tmpdir / "training"),
            },
            fileobj,
        )
    return load_config(str(filename), FilterModelPipeline)


@pytest.fixture
def setup_generated_data(shared_datadir):
    filename = shared_datadir / "big_dummy_generated_library.csv"
    with open(filename, "w") as fileobj:
        fileobj.write("reaction_smiles\treaction_hash\tclassification\tring_breaker\tretro_template\ttemplate_hash\n")


def test_create_lib(setup_config, setup_generated_data, shared_datadir):
    create_filter_lib(setup_config)

    assert (shared_datadir / "big_dummy_filter_library.csv").exists()
    assert (shared_datadir / "big_dummy_split_indices.npz").exists()

    data = pd.read_csv(shared_datadir / "big_dummy_filter_library.csv", sep="\t")
    assert len(data) == 10


def test_featurize(setup_config, setup_generated_data, shared_datadir):
    create_filter_lib(setup_config)
    featurize(setup_config)

    assert (shared_datadir / "big_dummy_inputs_prod.npz").exists()
    assert (shared_datadir / "big_dummy_inputs_rxn.npz").exists()
    assert (shared_datadir / "big_dummy_labels.npz").exists()

    data = np.load(shared_datadir / "big_dummy_labels.npz")["arr_0"]
    assert data.shape == (10,)

    data = sparse.load_npz(shared_datadir / "big_dummy_inputs_prod.npz")
    assert data.shape == (10, 2048)

    data = sparse.load_npz(shared_datadir / "big_dummy_inputs_rxn.npz")
    assert data.shape == (10, 2048)


def test_split_data(setup_config, setup_generated_data, shared_datadir):
    create_filter_lib(setup_config)
    featurize(setup_config)
    split_data(setup_config)

    assert (shared_datadir / "big_dummy_training_labels.npz").exists()
    assert (shared_datadir / "big_dummy_training_inputs_prod.npz").exists()
    assert (shared_datadir / "big_dummy_training_inputs_rxn.npz").exists()
    assert (shared_datadir / "big_dummy_validation_labels.npz").exists()
    assert (shared_datadir / "big_dummy_validation_inputs_prod.npz").exists()
    assert (shared_datadir / "big_dummy_validation_inputs_rxn.npz").exists()
    assert (shared_datadir / "big_dummy_testing_labels.npz").exists()
    assert (shared_datadir / "big_dummy_testing_inputs_rxn.npz").exists()
    assert (shared_datadir / "big_dummy_testing_inputs_prod.npz").exists()

    data = np.load(shared_datadir / "big_dummy_training_labels.npz")["arr_0"]
    assert data.shape == (6,)

    data = sparse.load_npz(shared_datadir / "big_dummy_training_inputs_prod.npz")
    assert data.shape == (6, 2048)

    data = sparse.load_npz(shared_datadir / "big_dummy_training_inputs_rxn.npz")
    assert data.shape == (6, 2048)

    data = np.load(shared_datadir / "big_dummy_validation_labels.npz")["arr_0"]
    assert data.shape == (2,)

    data = sparse.load_npz(shared_datadir / "big_dummy_validation_inputs_prod.npz")
    assert data.shape == (2, 2048)

    data = sparse.load_npz(shared_datadir / "big_dummy_validation_inputs_rxn.npz")
    assert data.shape == (2, 2048)

    data = np.load(shared_datadir / "big_dummy_testing_labels.npz")["arr_0"]
    assert data.shape == (2,)

    data = sparse.load_npz(shared_datadir / "big_dummy_testing_inputs_prod.npz")
    assert data.shape == (2, 2048)

    data = sparse.load_npz(shared_datadir / "big_dummy_testing_inputs_rxn.npz")
    assert data.shape == (2, 2048)
