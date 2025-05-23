"""Module containing the model interface common to all aizynthmodels."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from aizynthmodels.utils.trainer import build_trainer, instantiate_scorers

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple

    from omegaconf import DictConfig

    from aizynthmodels.utils.type_utils import StrDict


class ModelInterface:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """
        :param config: OmegaConf config loaded by hydra. Contains the input args of the model,
            including model checkpoint, model hyperparameters, input/output files, etc.

        The config includes the following arguments:
            # Trainer args
            seed: 1
            batch_size: 128
            n_devices (int): Number of GPUs to use.
            limit_val_batches: 1.0  # For training
            n_buckets: 12           # For training
            n_nodes: 1              # For training
            acc_batches: 1          # For training
            accelerator: null       # For training

            # Data args
            data_path (str): path to data used for training or inference
            dataset_part (str): Which dataset split to run inference on. ["full", "train", "val", "test"]
            task (str): the model task ["forward_prediction", "backward_prediction", "route_distances"]

            # Model args
            model_path (Optional[str]): Path to model weights.
            mode(str): Whether to train the model ("train") or use
                model for evaluations ("eval").
            device (str): Which device to run model and beam search on ("cuda" / "cpu").
            resume_training (bool): Whether to continue training from the supplied
                .ckpt file.

            # model-specific hyperparameters
            model_hyperparams:
                learning_rate (float): the learning rate (for training/fine-tuning)
                weight_decay (float): the weight decay (for training/fine-tuning)

            callbacks: list of Callbacks
            datamodule: the DataModule to use

            # Inference args
            scorers: list of Scores to evaluate predictions against ground-truth
        """

        self.config = config

        self.data_kwargs = {}
        self.mode = config.mode
        logging.info(f"mode: {self.mode}")

        self.device = self.config.get("device", "cuda")
        self.scores = instantiate_scorers(self.config.get("scorers"))

        if self.config.get("trainer"):
            self.trainer = build_trainer(config)

    def get_dataloader(self, dataset: str, datamodule: Optional[pl.LightningDataModule] = None) -> DataLoader:
        """
        Get the dataloader for a subset of the data from a specific datamodule.

        :param dataset: One in ["full", "train", "val", "test"].
                Specifies which part of the data to return.
        :param datamodule: pytorchlightning datamodule.
                If None -> Will use self.datamodule.
        :return: A dataloader
        """
        if dataset not in ["full", "train", "val", "test"]:
            raise ValueError(f"Unknown dataset : {dataset}. Should be either 'full', 'train', 'val' or 'test'.")

        if datamodule is None:
            datamodule = self.datamodule

        dataloader = getattr(datamodule, f"{dataset}_dataloader")()
        return dataloader

    def fit(self) -> None:
        """
        Fit model to training data in self.datamodule and using parameters specified in
        the trainer object.
        """
        self.model.to(self.device)
        self.trainer.fit(self.model, datamodule=self.datamodule)
        return

    def on_device(self, batch: StrDict) -> StrDict:
        """
        Move data in "batch" to the current model device.

        :param batch: batch input data to model.
        :return: batch data on current device.
        """
        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()
        }
        return device_batch

    def predict(
        self,
        dataset: Optional[str] = None,
        dataloader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> StrDict:
        """
        Predict output given dataloader, specified by 'dataset'.
        :param dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
        :param dataloader: If None -> dataloader
                will be retrieved from self.datamodule.
        :return: Dictionary with predictions (e.g. sampled smiles), logits (e.g. log-likelihoods)
            or probabilities, and  ground truth (e.g. target smiles)
        """

        output_keys = [
            "predictions",
            "logits",
            "probabilities",
            "log_likelihoods",
            "ground_truth",
        ]

        if dataloader is None:
            dataset_part = dataset if dataset else self.config.dataset_part
            dataloader = self.get_dataloader(dataset_part)

        self._prediction_setup(**kwargs)

        self.model.to(self.device)
        self.model.eval()

        predictions: StrDict = {}
        for batch in dataloader:
            batch = self.on_device(batch)
            with torch.no_grad():
                batch_output = self._predict_batch(batch)
            predictions = self._update_prediction_output(predictions, batch_output, output_keys)

        return predictions

    def score_model(
        self,
        dataset: Optional[str] = None,
        dataloader: Optional[DataLoader] = None,
        output_score_data: Optional[str] = None,
        output_predictions: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Score model performance on dataset in terms of accuracy (top-1 and top-K) and
        similarity of top-1 molecules. Also collects basic logging scores (loss, etc.).

        :param n_predictions: Number of predictions to generate.
        :param dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
        :param dataloader: If None -> dataloader will be
                retrieved from self.datamodule.
        :param output_score_data: Path to output .csv file with model performance. If None ->
                Will not write DataFrame to file.
        :param output_predictions: Path to output .json file with predictions.
                If None -> Will not write DataFrame to file.
        :return: Tuple with dataframe with calculated scores/metrics and dataframe with
                predictions and probabilities/log-likelihoods/logits
        """
        if not output_score_data:
            output_score_data = self.config.get("output_score_data")

        if not output_predictions:
            output_predictions = self.config.get("output_predictions")

        self._prediction_setup(**kwargs)
        for callback in self.trainer.callbacks:
            if hasattr(callback, "set_output_files"):
                callback.set_output_files(output_score_data, output_predictions)

        if dataloader is None:
            dataset_part = dataset if dataset else self.config.dataset_part
            dataloader = self.get_dataloader(dataset_part)

        self.model.to(self.device)
        self.model.eval()

        for b_idx, batch in enumerate(dataloader):
            batch = self.on_device(batch)
            metrics = self.model.test_step(batch, b_idx)
            self.model.test_step_outputs = []

            for callback in self.trainer.callbacks:
                if not isinstance(callback, pl.callbacks.progress.ProgressBar):
                    callback.on_test_batch_end(self.trainer, self.model, metrics, batch, b_idx, 0)

    def _prediction_setup(self, n_predictions: int = 1, **kwargs: Any) -> None:
        self.n_predictions = n_predictions

    def _predict_batch(self, batch: StrDict) -> StrDict:
        raise NotImplementedError("_predict_batch is not implemented for the interface class.")

    def _update_prediction_output(self, output: StrDict, batch_output: StrDict, output_keys: List[str]) -> StrDict:
        for key in output_keys:
            if key in batch_output and batch_output[key] is not None:
                if key not in output:
                    output[key] = []
                output[key].extend(batch_output[key])
        return output
