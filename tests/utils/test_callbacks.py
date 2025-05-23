import os

import hydra
import pandas as pd
import pytest
from omegaconf import OmegaConf

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.utils.callbacks import LearningRateMonitor, ScoreCallback, StepCheckpoint, ValidationScoreCallback
from aizynthmodels.utils.configs.chemformer import fine_tune as FineTune  # noqa: F401
from aizynthmodels.utils.configs.chemformer import inference_score as InferenceScore  # noqa: F401
from aizynthmodels.utils.trainer import instantiate_callbacks


@pytest.fixture
def train_config(vocabulary_path):

    model_hyperparams = {
        "d_model": 4,
        "max_seq_len": 50,
        "batch_first": False,
        "num_layers": 1,
        "num_heads": 2,
        "warm_up_steps": 2,
        "d_feedforward": 2,
        "dropout": 0.0,
    }

    config = OmegaConf.create(
        {
            "model_hyperparams": model_hyperparams,
            "n_predictions": 3,
            "datamodule": None,
            "vocabulary_path": vocabulary_path,
            "n_devices": 1,
            "device": "cpu",
            "mode": "eval",
            "task": "forward_prediction",
        }
    )
    return config


@pytest.fixture
def inference_config(vocabulary_path):

    model_hyperparams = {
        "max_seq_len": 50,
    }

    config = OmegaConf.create(
        {
            "model_hyperparams": model_hyperparams,
            "n_predictions": 3,
            "datamodule": None,
            "vocabulary_path": vocabulary_path,
            "n_devices": 1,
            "device": "cpu",
            "mode": "eval",
            "task": "forward_prediction",
        }
    )
    return config


@pytest.fixture
def ckpt_path(tmpdir, train_config, setup_synthesis_datamodule):
    with hydra.initialize(config_path=None):
        fine_tune_config = hydra.compose(config_name="fine_tune")
    fine_tune_config = OmegaConf.merge(fine_tune_config, train_config)

    fine_tune_config.n_epochs = 1
    fine_tune_config.acc_batches = 1
    fine_tune_config.output_directory = tmpdir
    fine_tune_config.logger.save_dir = tmpdir
    fine_tune_config.trainer.accelerator = "cpu"

    chemformer = Chemformer(fine_tune_config)
    chemformer.model.save_hyperparameters({"num_steps": 1})
    chemformer.datamodule = setup_synthesis_datamodule
    chemformer.fit()  # Run fit to save one dummy model ckpt
    return str(tmpdir / "forward_prediction" / "version_0")


def test_load_callbacks():
    callbacks = instantiate_callbacks(
        [
            "LearningRateMonitor",
            "ModelCheckpoint",
            {"StepCheckpoint": [{"step_interval": 1}]},
            "ValidationScoreCallback",
            "ScoreCallback",
        ]
    )
    assert len(callbacks.objects()) == 5


def test_step_checkpoint_wrong_input():
    with pytest.raises(TypeError, match="step_interval must be of type int, got type <class 'float'>"):
        instantiate_callbacks(
            [
                {"StepCheckpoint": [{"step_interval": 0.5}]},
            ]
        )


def test_learning_rate_monitor(tmpdir, train_config, setup_synthesis_datamodule):
    config = train_config
    config.callbacks = [{"LearningRateMonitor": [{"logging_interval": "step"}]}, "ValidationScoreCallback"]
    config.n_epochs = 3
    config.check_val_every_n_epoch = 1
    config.acc_batches = 1

    with hydra.initialize(config_path=None):
        fine_tune_config = hydra.compose(config_name="fine_tune")
    config = OmegaConf.merge(fine_tune_config, config)

    config.trainer.accelerator = "cpu"
    config.output_directory = tmpdir
    config.logger.save_dir = tmpdir
    chemformer = Chemformer(config)
    chemformer.model.save_hyperparameters({"num_steps": 3})
    chemformer.datamodule = setup_synthesis_datamodule

    assert isinstance(chemformer.trainer.callbacks[0], LearningRateMonitor)
    assert isinstance(chemformer.trainer.callbacks[1], ValidationScoreCallback)
    chemformer.fit()
    assert "logged_train_metrics.csv" in os.listdir(tmpdir / "forward_prediction" / "version_0")


def test_step_checkpoint(tmpdir, train_config, setup_synthesis_datamodule):
    config = train_config
    config.callbacks = [{"StepCheckpoint": [{"step_interval": 1}]}]

    with hydra.initialize(config_path=None):
        fine_tune_config = hydra.compose(config_name="fine_tune")
    config = OmegaConf.merge(fine_tune_config, config)

    config.n_epochs = 2
    config.acc_batches = 1
    config.output_directory = tmpdir
    config.logger.save_dir = tmpdir
    config.trainer.accelerator = "cpu"

    chemformer = Chemformer(config)
    chemformer.model.save_hyperparameters({"num_steps": 2})
    chemformer.datamodule = setup_synthesis_datamodule

    assert isinstance(chemformer.trainer.callbacks[0], StepCheckpoint)
    chemformer.fit()
    assert "step=1.ckpt" in os.listdir(tmpdir / "forward_prediction" / "version_0" / "checkpoints")


def test_score_callback(tmpdir, ckpt_path, inference_config, setup_synthesis_datamodule):
    output_score_data = tmpdir / "metrics_scores.csv"
    output_predictions = tmpdir / "predictions.json"

    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="inference_score")
    config = OmegaConf.merge(config, inference_config)

    config.model_path = ckpt_path + "/checkpoints/last.ckpt"
    config.output_score_data = output_score_data
    config.output_predictions = output_predictions
    config.n_predictions = 3
    config.device = "cpu"
    config.trainer.accelerator = "cpu"

    chemformer = Chemformer(config)
    chemformer.datamodule = setup_synthesis_datamodule
    assert isinstance(chemformer.trainer.callbacks[0], ScoreCallback)
    chemformer.score_model(dataset="test")

    metrics_data = pd.read_csv(config.output_score_data, sep="\t")
    assert "predictions.json" in os.listdir(tmpdir)
    assert "test_loss" in metrics_data.columns.values
    assert "accuracy_top_1" in metrics_data.columns.values
    assert metrics_data.shape[0] == 1
