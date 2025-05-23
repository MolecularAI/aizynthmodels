import os

import hydra
import pandas as pd
import pytest
import torch

from aizynthmodels.chemformer.models import BARTModel
from aizynthmodels.chemformer.sampler import SMILESSampler
from aizynthmodels.model_interface import ModelInterface
from aizynthmodels.utils.configs.chemformer import fine_tune as FineTune  # noqa: F401
from aizynthmodels.utils.configs.chemformer import inference_score as InferenceScore  # noqa: F401
from aizynthmodels.utils.configs.chemformer import predict as Predict  # noqa: F401


def next_batch(dataloader):
    return next(enumerate(dataloader))[1]


def targets_from_dataloader(dataloader):
    return next_batch(dataloader)["target_smiles"]


@pytest.fixture
def setup_bart(setup_basic_tokenizer):

    tokenizer = setup_basic_tokenizer

    model_hyperparams = {
        "vocabulary_size": len(tokenizer),
        "d_model": 4,
        "pad_token_idx": 1,
        "max_seq_len": 50,
        "n_predictions": 3,
        "batch_first": False,
        "num_layers": 1,
        "num_heads": 2,
        "num_steps": 4,
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

    sampler = SMILESSampler(
        tokenizer,
        model_hyperparams["max_seq_len"],
        device="cpu",
        sample_unique=False,
        sampler_node="BeamSearchSampler",
    )

    model = BARTModel(model_hyperparams)
    model.sampler = sampler
    model.n_predictions = 1
    return model


def test_model_interface_init():
    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    model = ModelInterface(config)
    assert isinstance(model, ModelInterface)
    assert model.device == "cpu"
    assert "top_k_accuracy" in model.scores.names()
    assert "fraction_invalid" in model.scores.names()


def test_get_dataloader(setup_synthesis_datamodule):
    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    datamodule = setup_synthesis_datamodule
    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    model = ModelInterface(config)

    with pytest.raises(
        ValueError, match="Unknown dataset : not-a-dataset. Should be either 'full', 'train', 'val' or 'test'"
    ):
        model.get_dataloader("not-a-dataset")

    train_targets = targets_from_dataloader(model.get_dataloader("train", datamodule))
    val_targets = targets_from_dataloader(model.get_dataloader("val", datamodule))
    test_targets = targets_from_dataloader(model.get_dataloader("test", datamodule))
    all_targets = targets_from_dataloader(model.get_dataloader("full", datamodule))

    assert len(train_targets) == 3
    assert len(val_targets) == 1
    assert len(test_targets) == 2
    assert len(all_targets) == 6

    model.datamodule = datamodule
    all_targets_internal_dm = targets_from_dataloader(model.get_dataloader("full"))
    assert all(tgt == tgt_internal for tgt, tgt_internal in zip(all_targets, all_targets_internal_dm))


def test_fit(setup_synthesis_datamodule, setup_bart, tmpdir):

    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    config.n_epochs = 2
    config.acc_batches = 1
    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    config.output_directory = tmpdir / "chemformer"

    datamodule = setup_synthesis_datamodule

    model = ModelInterface(config)
    model.datamodule = datamodule

    model.model = setup_bart
    model.model.scores = model.scores
    model.fit()

    assert "logged_train_metrics.csv" in os.listdir(tmpdir / "chemformer" / "backward_prediction" / "version_0")
    logged_data = pd.read_csv(
        tmpdir / "chemformer" / "backward_prediction" / "version_0" / "logged_train_metrics.csv", sep="\t"
    )
    assert "accuracy_top_1" in logged_data.columns.values
    assert "fraction_invalid" in logged_data.columns.values


def test_on_device(setup_synthesis_datamodule):

    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    datamodule = setup_synthesis_datamodule

    model = ModelInterface(config)
    dataloader = model.get_dataloader("train", datamodule)

    batch = next_batch(dataloader)
    batch = model.on_device(batch)
    for x in batch.values():
        if isinstance(x, torch.Tensor):
            assert str(x.device) == model.device


def test_predict(setup_synthesis_datamodule, setup_bart):
    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="predict")

    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    datamodule = setup_synthesis_datamodule
    model = ModelInterface(config)
    dataloader = model.get_dataloader("test", datamodule)

    with pytest.raises(AttributeError, match="object has no attribute 'model'"):
        model.predict(dataloader=dataloader)

    model.model = setup_bart
    with pytest.raises(NotImplementedError, match="_predict_batch is not implemented for the interface class"):
        model.predict(dataloader=dataloader)

    with pytest.raises(AttributeError, match="object has no attribute 'datamodule'"):
        model.predict(dataset="test")

    model.datamodule = datamodule
    with pytest.raises(NotImplementedError, match="_predict_batch is not implemented for the interface class"):
        model.predict(dataset="test")


def test_prediction_output(setup_synthesis_datamodule, setup_bart):
    output_keys = [
        "predictions",
        "logits",
        "probabilities",
        "log_likelihoods",
        "ground_truth",
    ]
    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="predict")

    config.device = "cpu"

    datamodule = setup_synthesis_datamodule

    model = ModelInterface(config)
    model.model = setup_bart
    dataloader = model.get_dataloader("train", datamodule)

    batch = next_batch(dataloader)
    smiles_batch, log_lhs_batch = model.model.sample_predictions(batch)

    if batch.get("ground_truth") is not None:
        batch_output = {
            "predictions": smiles_batch,
            "log_likelihoods": log_lhs_batch,
            "ground_truth": batch.get("ground_truth"),
        }
    else:
        batch_output = {
            "predictions": smiles_batch,
            "log_likelihoods": log_lhs_batch,
        }

    predictions = {}
    predictions = model._update_prediction_output(predictions, batch_output, output_keys)
    assert predictions == batch_output
    predictions = model._update_prediction_output(predictions, batch_output, output_keys)
    assert len(predictions["predictions"]) == 2 * len(batch_output["predictions"])


def test_score_model(tmpdir, setup_bart, setup_synthesis_datamodule):
    output_score_data = tmpdir / "metrics_scores.csv"
    output_predictions = tmpdir / "sampled_predictions.json"

    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="inference_score")

    config.n_predictions = 3
    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    config.output_score_data = output_score_data
    config.output_predictions = output_predictions

    datamodule = setup_synthesis_datamodule

    model = ModelInterface(config)
    model.model = setup_bart
    model.model.scores = model.scores
    model.datamodule = datamodule

    model.score_model(dataloader=model.get_dataloader("test", datamodule))
    predictions1 = pd.read_json(output_predictions, orient="table")

    output_predictions = tmpdir / "sampled_predictions2.json"
    model.config.output_predictions = output_predictions
    model.trainer.callbacks[0]._metrics = pd.DataFrame()
    model.trainer.callbacks[0]._predictions = pd.DataFrame()

    model.score_model(dataset="test")
    predictions2 = pd.read_json(output_predictions, orient="table")

    pd.testing.assert_frame_equal(predictions1, predictions2)
