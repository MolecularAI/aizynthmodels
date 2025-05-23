from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
from pytorch_lightning.callbacks import Callback


class LearningRateMonitor(plc.LearningRateMonitor):
    callback_name = "LearningRateMonitor"

    def __init__(self, logging_interval: str = "step", log_momentum: bool = False, **kwargs: Any) -> None:
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum, **kwargs)

    def __repr__(self):
        return self.callback_name


class ModelCheckpoint(plc.ModelCheckpoint):
    callback_name = "ModelCheckpoint"

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: str = "validation_loss",
        verbose: bool = False,
        save_last: bool = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[int] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
            **kwargs,
        )

    def __repr__(self):
        return self.callback_name


class StepCheckpoint(Callback):
    callback_name = "StepCheckpoint"

    def __init__(self, step_interval: int = 50000) -> None:
        super().__init__()

        if not isinstance(step_interval, int):
            raise TypeError(f"step_interval must be of type int, got type {type(step_interval)}")

        self.step_interval = step_interval

    def __repr__(self):
        return self.callback_name

    # def on_batch_end(self, trainer, model):
    # Ideally this should on_after_optimizer_step, but that isn't available in pytorch lightning (yet?)
    def on_after_backward(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        step = trainer.global_step
        if (step != 0) and (step % self.step_interval == 0):
            # if (step % self.step_interval == 0):
            self._save_model(trainer, model, step)

    def _save_model(self, trainer: pl.Trainer, model: pl.LightningModule, step: int) -> None:
        if trainer.logger is not None:
            if trainer.logger.log_dir is not None:
                data_path = trainer.logger.log_dir
            elif trainer.logger.save_dir is not None:
                data_path = trainer.logger.save_dir
            else:
                data_path = None
        else:
            data_path = trainer.ckpt_path
        ckpt_path = os.path.join(data_path, "checkpoints")

        save_path = f"{ckpt_path}/step={str(step)}.ckpt"
        logging.info(f"Saving step checkpoint in {save_path}")
        trainer.save_checkpoint(save_path)


class ValidationScoreCallback(Callback):
    """
    Retrieving scores from the validation epochs and write to file continuously.
    """

    callback_name = "ValidationScoreCallback"

    def __init__(self) -> None:
        super().__init__()
        self._metrics = pd.DataFrame()
        self._skip_logging = True

    def __repr__(self):
        return self.callback_name

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._skip_logging:
            self._skip_logging = False
            return

        logged_metrics = {
            key: [val.to(torch.device("cpu")).numpy()]
            for key, val in trainer.callback_metrics.items()
            if key != "mol_acc"
        }

        metrics = {"epoch": pl_module.current_epoch}
        metrics.update(logged_metrics)
        metrics_df = pd.DataFrame(metrics)

        self._metrics = pd.concat([self._metrics, metrics_df], axis=0, ignore_index=True)

        self.out_directory = self._get_out_directory(trainer)
        self._save_logged_data()
        return

    def _get_out_directory(self, trainer: pl.Trainer) -> str:
        if trainer.logger is not None:
            if trainer.logger.log_dir is not None:
                data_path = trainer.logger.log_dir
            elif trainer.logger.save_dir is not None:
                data_path = trainer.logger.save_dir
            else:
                data_path = None
        else:
            data_path = trainer.ckpt_path
        return data_path

    def _save_logged_data(self) -> None:
        """
        Retrieve and write data (model validation) logged during training.
        """
        outfile = self.out_directory + "/logged_train_metrics.csv"
        self._metrics.to_csv(outfile, sep="\t", index=False)
        logging.info("Logged training/validation set loss written to: " + outfile)
        return


class ScoreCallback(Callback):
    """
    Retrieving scores from test step and write to file continuously.
    """

    callback_name = "ScoreCallback"

    def __init__(
        self,
        output_scores: str = "metrics_scores.csv",
        output_predictions: str = "predictions.json",
    ) -> None:
        super().__init__()
        self._metrics = pd.DataFrame()
        self._predictions = pd.DataFrame()

        self._metrics_output = output_scores
        self._predictions_output = output_predictions

    def __repr__(self):
        return self.callback_name

    def set_output_files(self, output_score_data: str, output_predictions: Optional[str]) -> None:
        self._metrics_output = output_score_data
        self._predictions_output = output_predictions

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        test_output: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        prediction_keys = [
            "ground_truth",
            "predictions",
            "target_smiles",
            "sampled_molecules",
            "sampled_molecules(unique)",
            "log_lhs",
            "logits",
            "probabilities",
            "log_lhs(unique)",
        ]

        logged_metrics = {key: [val] for key, val in test_output.items() if key not in prediction_keys}

        for key, val in logged_metrics.items():
            if isinstance(val[0], torch.Tensor):
                logged_metrics[key] = [val[0].cpu().detach().numpy()]

        predictions = {key: [val] for key, val in test_output.items() if key in prediction_keys}

        metrics_df = pd.DataFrame(logged_metrics)
        predictions_df = pd.DataFrame(predictions)

        self._metrics = pd.concat([self._metrics, metrics_df], axis=0, ignore_index=True)
        self._predictions = pd.concat([self._predictions, predictions_df], axis=0, ignore_index=True)

        self._save_logged_data()

    def _save_logged_data(self) -> None:
        """
        Retrieve and write data (model validation) logged during training.
        """
        self._metrics.to_csv(self._metrics_output, sep="\t", index=False)
        logging.info("Test set metrics written to file: " + self._metrics_output)

        if self._predictions_output:
            self._predictions.to_json(self._predictions_output, orient="table")
            logging.info("Test set predictions written to file: " + self._predictions_output)
        return
