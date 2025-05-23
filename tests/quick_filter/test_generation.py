import pandas as pd
import pytest
import yaml

from aizynthmodels.quick_filter.tools.generate_data import main as generate_data
from aizynthmodels.utils.configs.quick_filter.pipelines import FilterModelPipeline
from aizynthmodels.utils.hydra import load_config


@pytest.fixture
def setup_config(shared_datadir, tmpdir):
    filename = tmpdir / "config.yaml"
    with open(filename, "w") as fileobj:
        yaml.dump(
            {
                "file_prefix": str(shared_datadir / "small_dummy"),
                "n_batches": 1,
                "training_fraction": 0.6,
                "output_directory": str(tmpdir / "training"),
            },
            fileobj,
        )
    return load_config(str(filename), FilterModelPipeline)


def test_generate_no_data(shared_datadir, setup_config):
    config = setup_config
    config.negative_generation.type = []

    generate_data(config)

    assert (shared_datadir / "small_dummy_generated_library.csv").exists()

    lib = pd.read_csv(shared_datadir / "small_dummy_generated_library.csv", sep="\t")
    assert len(lib) == 0


def test_generate_strict_appl(shared_datadir, setup_config):
    config = setup_config
    config.negative_generation.type = ["strict"]

    generate_data(config)

    assert (shared_datadir / "small_dummy_generated_library.csv").exists()

    lib = pd.read_csv(shared_datadir / "small_dummy_generated_library.csv", sep="\t")
    assert len(lib) == 2

    reaction_hashes = lib["reaction_hash"]
    # Generated reaction #1 and #2 have the same reactants but different products
    assert reaction_hashes[0].split(">>")[0] == reaction_hashes[1].split(">>")[0]
    assert reaction_hashes[0].split(">>")[1] == "AMMCLTLLHBZPMU-UHFFFAOYSA-N"
    assert reaction_hashes[1].split(">>")[1] == "FPTYNIYEAIRHSG-UHFFFAOYSA-N"


def test_generate_random_appl(shared_datadir, setup_config):
    config = setup_config
    config.negative_generation.type = ["random"]

    generate_data(config)

    assert (shared_datadir / "small_dummy_generated_library.csv").exists()

    templ_lib = pd.read_csv(shared_datadir / "small_dummy_template_library.csv", sep="\t")
    lib = pd.read_csv(shared_datadir / "small_dummy_generated_library.csv", sep="\t")
    assert len(lib) == 2

    # Generated reaction #1 is the same reactants as original reaction #1, but with differen products
    assert lib["reaction_hash"][0].split(">>")[0] == templ_lib["reaction_hash"][0].split(">>")[0]
    assert lib["reaction_hash"][0].split(">>")[1] != templ_lib["reaction_hash"][0].split(">>")[1]

    # Generated reaction #2 is the same reactants as original reaction #2, but with differen products
    assert lib["reaction_hash"][1].split(">>")[0] == templ_lib["reaction_hash"][1].split(">>")[0]
    assert lib["reaction_hash"][1].split(">>")[1] != templ_lib["reaction_hash"][1].split(">>")[1]
