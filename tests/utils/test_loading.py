import pytest
from omegaconf import OmegaConf

from aizynthmodels.chemformer.data import SynthesisDataModule
from aizynthmodels.chemformer.data.datamodule import __name__ as data_module
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
from aizynthmodels.chemformer.models import BARTModel
from aizynthmodels.chemformer.models import __name__ as models_module
from aizynthmodels.utils.loading import build_datamodule, build_model, load_dynamic_class, load_item


@pytest.fixture
def bart_hyperparams():
    model_hyperparams = {
        "d_model": 4,
        "pad_token_idx": 1,
        "max_seq_len": 32,
        "n_predictions": 3,
        "batch_first": True,
        "num_layers": 1,
        "num_heads": 2,
        "num_steps": 2,
        "warm_up_steps": 2,
        "d_feedforward": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "optimizer": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "scheduler": "cycle",
            "warm_up_steps": 8000,
            "betas": [0.9, 0.999],
        },
    }
    return OmegaConf.create(model_hyperparams)


@pytest.mark.parametrize(
    ("item", "module"),
    [
        ("BARTModel", models_module),
        ({"BARTModel": [{"ckpt_path": "dummy"}]}, models_module),
        ({"BARTModel": None}, models_module),
        ({"aizynthmodels.chemformer.models.bart.BARTModel": None}, ""),
    ],
)
def test_load_item(item, module):
    item = load_item(item, module)


def test_load_item_not_in_module():
    with pytest.raises(ValueError):
        load_item("BARTModel", data_module)


def test_load_dynamic_class_error():
    with pytest.raises(ValueError, match="Must provide default_module argument if not given in name_spec"):
        load_dynamic_class("SomeClass")

    with pytest.raises(ValueError, match="Unable to load module:"):
        load_dynamic_class("not.existing.SomeClass")


@pytest.mark.parametrize(
    ("reactant_smiles", "product_smiles", "reverse", "target_smiles"),
    [
        ("Cl.c1ccccc1", "Clc1ccccc1", False, "Clc1ccccc1"),
        ("Cl.c1ccccc1", "Clc1ccccc1", True, "Cl.c1ccccc1"),
    ],
)
def test_build_datamodule(reactant_smiles, product_smiles, reverse, target_smiles):

    datamodule = build_datamodule(
        {
            "type": "SynthesisDataModule",
            "arguments": [
                {"reactants": [reactant_smiles]},
                {"products": [product_smiles]},
                {"batch_size": 1},
                {"dataset_path": ""},
                {"tokenizer": ChemformerTokenizer(filename="tests/chemformer/data/simple_vocab.json")},
                {"max_seq_len": 512},
                {"reverse": reverse},
            ],
        },
        data_module,
    )

    assert isinstance(datamodule, SynthesisDataModule)
    assert len(datamodule.full_dataloader()) == 1
    batch = [data for data in datamodule.full_dataloader()][0]
    assert len(batch["target_smiles"]) == 1
    assert batch["target_smiles"][0] == target_smiles


def test_build_model(bart_hyperparams):
    model = build_model(
        {"type": "BARTModel", "arguments": [{"vocabulary_size": 200}]}, models_module, bart_hyperparams, mode="eval"
    )
    assert isinstance(model, BARTModel)

    with pytest.raises(FileNotFoundError):
        build_model(
            {"type": "BARTModel", "arguments": [{"ckpt_path": "dummpy-path", "vocabulary_size": 200}]},
            models_module,
            bart_hyperparams,
            mode="eval",
        )
