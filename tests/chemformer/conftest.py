import numpy as np
import omegaconf as oc
import pandas as pd
import pytest
import pytorch_lightning as pl

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.chemformer.data import SynthesisDataModule
from aizynthmodels.chemformer.data.encoder import BatchEncoder
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer, ReplaceTokensMasker, SpanTokensMasker
from aizynthmodels.chemformer.utils.defaults import DEFAULT_MAX_SEQ_LEN

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer.fine_tune import FineTune
from aizynthmodels.utils.configs.chemformer.round_trip_inference import RoundTripInference
from aizynthmodels.utils.trainer import instantiate_callbacks


@pytest.fixture
def reactant_data():
    return ["CCO.C", "CCCl", "C(=O)CBr"]


@pytest.fixture
def product_data():
    return ["cc", "CCl", "CBr"]


@pytest.fixture
def example_tokens():
    return [
        ["^", "C", "(", "=", "O", ")", "unknown", "&"],
        ["^", "C", "C", "<SEP>", "C", "Br", "&"],
    ]


@pytest.fixture
def regex_tokens():
    regex = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    return regex.split("|")


@pytest.fixture
def smiles_data():
    return ["CCO.Ccc", "CCClCCl", "C(=O)CBr"]


@pytest.fixture
def mock_random_choice(mocker):
    class ToggleBool:
        def __init__(self):
            self.state = True

        def __call__(self, *args, **kwargs):
            states = []
            for _ in range(kwargs["k"]):
                states.append(self.state)
                self.state = not self.state
            return states

    mocker.patch(
        "aizynthmodels.chemformer.data.tokenizer.random.choices",
        side_effect=ToggleBool(),
    )


@pytest.fixture
def setup_tokenizer(regex_tokens, smiles_data):
    def wrapper(tokens=None):
        return ChemformerTokenizer(smiles=smiles_data, tokens=tokens, regex_token_patterns=regex_tokens)

    return wrapper


@pytest.fixture
def setup_encoder(reactant_data, product_data, default_config, regex_tokens):
    tokenizer = ChemformerTokenizer(
        smiles=reactant_data + product_data,
        regex_token_patterns=regex_tokens,
    )
    masker = ReplaceTokensMasker(tokenizer)
    encoder = BatchEncoder(
        tokenizer,
        masker=masker,
        max_seq_len=default_config.model_hyperparams["max_seq_len"],
    )
    return tokenizer, masker, encoder


@pytest.fixture
def setup_masker(setup_tokenizer):
    def wrapper(cls=SpanTokensMasker, kwargs={}):
        tokenizer = setup_tokenizer()
        return tokenizer, cls(tokenizer, **kwargs)

    return wrapper


@pytest.fixture
def round_trip_params(shared_datadir):
    params = {
        "n_samples": 3,
        "beam_size": 5,
        "batch_size": 2,
        "round_trip_input_data": shared_datadir / "round_trip_input_data.csv",
    }
    return params


@pytest.fixture
def round_trip_raw_prediction_data(shared_datadir):
    round_trip_df = pd.read_json(shared_datadir / "round_trip_predictions_raw.json", orient="table")
    round_trip_predictions = [np.array(smiles_lst) for smiles_lst in round_trip_df["round_trip_smiles"].values]
    data = {
        "sampled_smiles": round_trip_predictions,
        "target_smiles": round_trip_df["target_smiles"].values,
    }
    return data


@pytest.fixture
def round_trip_converted_prediction_data(shared_datadir):
    round_trip_df = pd.read_json(shared_datadir / "round_trip_predictions_converted.json", orient="table")
    round_trip_predictions = [np.array(smiles_lst) for smiles_lst in round_trip_df["round_trip_smiles"].values]
    data = {
        "sampled_smiles": round_trip_predictions,
        "target_smiles": round_trip_df["target_smiles"].values,
    }
    return data


@pytest.fixture
def default_config(shared_datadir):

    model_hyperparams = {
        "d_model": 4,
        "pad_token_idx": 1,
        "max_seq_len": DEFAULT_MAX_SEQ_LEN,
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
        "num_hidden_nodes": [256],
        "num_classes": 3,
    }

    trainer_args = {
        "accelerator": "cpu",
        "devices": 1,
        "strategy": "auto",
        "precision": "16-mixed",
        "deterministic": True,
    }

    config = oc.OmegaConf.create(
        {
            "model_hyperparams": model_hyperparams,
            "input_data": shared_datadir / "example_data_uspto.csv",
            "backward_predictions": shared_datadir / "example_data_backward_sampled_smiles_uspto50k.json",
            "output_score_data": "temp_metrics.csv",
            "dataset_part": "test",
            "n_predictions": 3,
            "batch_size": 3,
            "sampler": "BeamSearchSampler",
            "datamodule": None,
            "vocabulary_path": shared_datadir / "simple_vocab.json",
            "n_devices": 1,
            "device": "cpu",
            "mode": "eval",
            "task": "forward_prediction",
            "target_column": "products",
            "model": {"type": "BARTModel", "arguments": [{"ckpt_path": None}]},
            "scorers": ["TopKAccuracyScore"],
            "callbacks": ["ScoreCallback"],
            "trainer_args": trainer_args,
        }
    )
    return config


@pytest.fixture
def model_batch_setup(default_config):
    pl.seed_everything(1)

    data = pd.read_csv(default_config.input_data, sep="\t")
    config = default_config.copy()

    config.model_hyperparams["batch_first"] = False
    chemformer = Chemformer(config)
    callbacks = instantiate_callbacks(config.callbacks)
    chemformer.trainer = pl.Trainer(**config.trainer_args, callbacks=callbacks.objects())

    datamodule = SynthesisDataModule(
        reactants=data["reactants"].values,
        products=data["products"].values,
        dataset_path="",
        tokenizer=chemformer.tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.model_hyperparams["max_seq_len"],
        reverse=False,
    )

    datamodule.setup()
    dataloader = datamodule.full_dataloader()
    batch_idx, batch_input = next(enumerate(dataloader))

    output_data = {
        "chemformer": chemformer,
        "batch_idx": batch_idx,
        "batch_input": batch_input,
    }
    return output_data


@pytest.fixture
def model_batch_setup_batch_first(default_config):
    config = default_config.copy()
    config.model_hyperparams["batch_first"] = True
    data = pd.read_csv(config.input_data, sep="\t")
    chemformer = Chemformer(config)

    datamodule = SynthesisDataModule(
        reactants=data["reactants"].values,
        products=data["products"].values,
        dataset_path="",
        tokenizer=chemformer.tokenizer,
        batch_size=config.batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        reverse=False,
    )

    datamodule.setup()
    dataloader = datamodule.full_dataloader()
    batch_idx, batch_input = next(enumerate(dataloader))

    output_data = {
        "chemformer": chemformer,
        "batch_idx": batch_idx,
        "batch_input": batch_input,
    }
    return output_data
