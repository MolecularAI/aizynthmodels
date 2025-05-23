from pathlib import Path

import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf
from scipy import sparse

from aizynthmodels.template_based.pipelines.scripts.create_template_lib import main as create_template_lib
from aizynthmodels.template_based.pipelines.scripts.featurize import main as featurize
from aizynthmodels.template_based.pipelines.scripts.split_data import main as split_data
from aizynthmodels.template_based.tools.extract_templates import main as extract_templates
from aizynthmodels.template_based.tools.stereo_flipper import main as stereo_flipper
from aizynthmodels.utils.configs.template_based.extract_templates import ExtractTemplates
from aizynthmodels.utils.configs.template_based.pipelines import ExpansionModelPipeline
from aizynthmodels.utils.configs.template_based.stereo_flipper import StereoFlipper
from aizynthmodels.utils.hydra import load_config


@pytest.fixture
def setup_config(shared_datadir, tmpdir):
    filename = tmpdir / "config.yaml"
    with open(filename, "w") as fileobj:
        yaml.dump(
            {
                "file_prefix": str(shared_datadir / "dummy"),
                "training_pipeline": {
                    "n_batches": 1,
                    "training_fraction": 0.6,
                    "training_output": str(tmpdir / "training"),
                    "selected_ids_path": None,
                    "routes_to_exclude": [str(shared_datadir / "dummy_ref_routes.json")],
                },
            },
            fileobj,
        )
    return load_config(str(filename), ExpansionModelPipeline)


def test_create_metadata(setup_config, shared_datadir):
    create_template_lib(setup_config)

    assert (shared_datadir / "dummy_template_code.csv").exists()
    assert (shared_datadir / "dummy_lookup.json").exists()
    assert (shared_datadir / "dummy_unique_templates.csv.gz").exists()
    assert (shared_datadir / "dummy_split_indices.npz").exists()

    data = pd.read_csv(shared_datadir / "dummy_template_code.csv")
    assert len(data) == 10

    data = pd.read_csv(shared_datadir / "dummy_unique_templates.csv.gz")
    assert len(data) == 8


def test_featurize(setup_config, shared_datadir):
    create_template_lib(setup_config)
    featurize(setup_config)

    assert (shared_datadir / "dummy_labels.npz").exists()
    assert (shared_datadir / "dummy_inputs.npz").exists()

    data = sparse.load_npz(shared_datadir / "dummy_labels.npz")
    assert data.shape == (10, 8)

    data = sparse.load_npz(shared_datadir / "dummy_inputs.npz")
    assert data.shape == (10, 2048)


def test_split_data(setup_config, shared_datadir):
    create_template_lib(setup_config)
    featurize(setup_config)
    split_data(setup_config)

    assert (shared_datadir / "dummy_training_labels.npz").exists()
    assert (shared_datadir / "dummy_training_inputs.npz").exists()
    assert (shared_datadir / "dummy_validation_labels.npz").exists()
    assert (shared_datadir / "dummy_validation_inputs.npz").exists()
    assert (shared_datadir / "dummy_testing_labels.npz").exists()
    assert (shared_datadir / "dummy_testing_inputs.npz").exists()

    data = sparse.load_npz(shared_datadir / "dummy_training_labels.npz")
    assert data.shape == (8, 8)

    data = sparse.load_npz(shared_datadir / "dummy_training_inputs.npz")
    assert data.shape == (8, 2048)

    data = sparse.load_npz(shared_datadir / "dummy_validation_labels.npz")
    assert data.shape == (1, 8)

    data = sparse.load_npz(shared_datadir / "dummy_validation_inputs.npz")
    assert data.shape == (1, 2048)

    data = sparse.load_npz(shared_datadir / "dummy_testing_labels.npz")
    assert data.shape == (1, 8)

    data = sparse.load_npz(shared_datadir / "dummy_testing_inputs.npz")
    assert data.shape == (1, 2048)


def test_extract_templates(shared_datadir, tmpdir):
    config = OmegaConf.structured(ExtractTemplates)

    config.radius = 1
    config.input_data = f"{shared_datadir}/dummy_template_library.csv"
    config.output_data = f"{tmpdir}/extracted_templates.csv"
    config.smiles_column = "reaction_smiles"
    config.ringbreaker_column = "ring_breaker"

    extract_templates(config)

    assert Path(config.output_data).exists()
    templates = pd.read_csv(config.output_data, sep="\t")
    assert templates["RetroTemplate"].equals(templates["retro_template"])
    assert templates["TemplateHash"].equals(templates["template_hash"])


def test_extract_templates_exception_handling(shared_datadir, tmpdir):
    config = OmegaConf.structured(ExtractTemplates)

    input_data = tmpdir / "input_data.csv"
    data = pd.read_csv(f"{shared_datadir}/dummy_template_library.csv", sep="\t")
    rsmi = data["reaction_smiles"].values
    rsmi[0] = "not-a-rxn-smiles"
    data["reaction_smiles"] = rsmi
    data.to_csv(input_data, sep="\t", index=False)

    config.radius = 1
    config.input_data = input_data
    config.output_data = f"{tmpdir}/extracted_templates.csv"
    config.smiles_column = "reaction_smiles"
    config.ringbreaker_column = "ring_breaker"

    with pytest.raises(ValueError, match="Expected 3 reaction components but got 1"):
        extract_templates(config)

    rsmi[0] = "not-a-smiles>>not-a-smiles"
    data["reaction_smiles"] = rsmi
    data.to_csv(input_data, sep="\t", index=False)
    extract_templates(config)

    assert Path(config.output_data).exists()
    templates = pd.read_csv(config.output_data, sep="\t")
    assert not templates["RetroTemplate"].equals(templates["retro_template"])


def test_stereo_flipper(tmpdir):
    config = OmegaConf.structured(StereoFlipper)

    df = pd.DataFrame(
        {
            "retro_template": [
                "[C@:2](=[O;D1;H0:3])-[C:5]>>[C@:2](=[O;D1;H0:3]).[C:5]-[OH;D1;+0:6]",
                "[C:2](=[O;D1;H0:3])-[C@@:5]>>[C:2](=[O;D1;H0:3]).[C@@:5]-[OH;D1;+0:6]",
                "[C@:2](=[O;D1;H0:3])-[C@@:5]>>[C@:2](=[O;D1;H0:3]).[C:5]-[OH;D1;+0:6]",
            ],
            "template_hash": ["th1", "th2", "th3"],
        }
    )

    df_expected = pd.DataFrame(
        {
            "retro_template": [
                "[C@:2](=[O;D1;H0:3])-[C:5]>>[C@:2](=[O;D1;H0:3]).[C:5]-[OH;D1;+0:6]",
                "[C:2](=[O;D1;H0:3])-[C@@:5]>>[C:2](=[O;D1;H0:3]).[C@@:5]-[OH;D1;+0:6]",
                "[C@:2](=[O;D1;H0:3])-[C@@:5]>>[C@:2](=[O;D1;H0:3]).[C:5]-[OH;D1;+0:6]",
                "[C@@:2](=[O;D1;H0:3])-[C:5]>>[C@@:2](=[O;D1;H0:3]).[C:5]-[OH;D1;+0:6]",
                "[C:2](=[O;D1;H0:3])-[C@:5]>>[C:2](=[O;D1;H0:3]).[C@:5]-[OH;D1;+0:6]",
            ],
            "template_hash": [
                "th1",
                "th2",
                "th3",
                "8f8d039267bd729d40596a1289c05c5650f20725c52a334b8f2c0861a2050d12",
                "44be6831639c0d7f3ba749bf2873d9ca4527293f798b266922ca10aec53911e5",
            ],
            "FlippedStereo": [False, False, False, True, True],
        }
    )

    config.input_data = f"{tmpdir}/retro_templates.csv"
    config.output_data = f"{tmpdir}/retro_templates.csv"
    config.query = "retro_template==retro_template"  # Select all
    config.template_column = "retro_template"
    config.template_hash_column = "template_hash"

    df.to_csv(config.input_data, sep="\t", index=False)

    stereo_flipper(config)

    df_out = pd.read_csv(config.output_data, sep="\t")
    pd.testing.assert_frame_equal(df_out, df_expected)
