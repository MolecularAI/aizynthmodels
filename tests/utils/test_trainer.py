import hydra
import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy

from aizynthmodels.chemformer.data import SynthesisDataModule
from aizynthmodels.utils.callbacks import CallbackCollection
from aizynthmodels.utils.configs.chemformer import fine_tune as FineTune  # noqa: F401
from aizynthmodels.utils.trainer import build_trainer, calc_train_steps, instantiate_callbacks, instantiate_logger


def test_instantiate_callbacks():
    callback_names = ["ScoreCallback", "ValidationScoreCallback"]
    callbacks = instantiate_callbacks(callback_names)

    assert isinstance(callbacks, CallbackCollection)
    assert len(callbacks.objects()) == 2
    assert all(str(cb) == name for cb, name in zip(callbacks.objects(), callback_names))

    callbacks = instantiate_callbacks(None)

    assert isinstance(callbacks, CallbackCollection)
    assert len(callbacks.objects()) == 0


def test_instantiate_logger():

    logger = instantiate_logger(None)
    assert logger == []

    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    logger = instantiate_logger(config.logger)
    assert isinstance(logger, TensorBoardLogger)

    with pytest.raises(TypeError, match="Logger config must be a DictConfig"):
        instantiate_logger(["some-logger"])


def test_calc_train_steps(seq2seq_data, setup_basic_tokenizer, setup_synthesis_datamodule):
    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    config.acc_batches = 1
    num_steps = calc_train_steps(config, setup_synthesis_datamodule)
    assert num_steps == 50

    config.acc_batches = 1
    datamodule = SynthesisDataModule(
        dataset_path=seq2seq_data, max_seq_len=128, tokenizer=setup_basic_tokenizer, batch_size=1
    )
    datamodule.setup()
    num_steps = calc_train_steps(config, datamodule)
    assert num_steps == 150


def test_build_trainer():
    with hydra.initialize(config_path=None):
        config = hydra.compose(config_name="fine_tune")

    trainer = build_trainer(config)
    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
