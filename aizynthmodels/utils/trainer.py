import logging
import math
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import TensorBoardLogger

from aizynthmodels.utils.callbacks import CallbackCollection
from aizynthmodels.utils.scores import ScoreCollection


def instantiate_callbacks(callbacks_config: Optional[ListConfig]) -> CallbackCollection:
    """Instantiates callbacks from config."""
    callbacks = CallbackCollection()

    if not callbacks_config:
        logging.info("No callbacks configs found! Skipping...")
        return callbacks

    callbacks.load_from_config(callbacks_config)
    return callbacks


def instantiate_scorers(scorer_config: Optional[ListConfig]) -> ScoreCollection:
    """Instantiates scorer from config."""

    scorer = ScoreCollection()
    if not scorer_config:
        logging.info("No scorer configs found! Skipping...")
        return scorer

    scorer.load_from_config(scorer_config)
    return scorer


def instantiate_logger(logger_config: Optional[DictConfig]) -> TensorBoardLogger:
    """Instantiates logger from config."""
    logger: TensorBoardLogger = []

    if not logger_config:
        logging.info("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    if isinstance(logger_config, DictConfig) and "_target_" in logger_config:
        logging.info(f"Instantiating logger <{logger_config._target_}>")
        logger = hydra.utils.instantiate(logger_config)

    return logger


def calc_train_steps(config: DictConfig, datamodule: pl.LightningDataModule) -> float:
    n_devices = config.n_devices
    batches_per_device = math.ceil(len(datamodule.train_dataloader()) / float(n_devices))
    train_steps = math.ceil(batches_per_device / config.acc_batches) * config.n_epochs
    return train_steps


def build_trainer(config: DictConfig) -> pl.Trainer:

    logging.info("Instantiating loggers...")
    logger = instantiate_logger(config.get("logger"))

    logging.info("Instantiating callbacks...")
    callbacks: CallbackCollection = instantiate_callbacks(config.get("callbacks"))

    if config.n_devices > 1:
        config.trainer.strategy = "ddp"

    logging.info("Building trainer...")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks.objects(), logger=logger)
    logging.info("Finished trainer.")

    logging.info(f"Default logging and checkpointing directory: {trainer.default_root_dir} or {trainer.ckpt_path}")
    return trainer
