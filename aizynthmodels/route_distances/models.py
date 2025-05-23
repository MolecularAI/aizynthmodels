""" Module containing the LSTM-based model for calculation route distances """

import logging
from typing import Any, List, Optional, Tuple

import pytorch_lightning as lightning
import torch
from omegaconf import DictConfig
from treelstm import TreeLSTM as TreeLSTMBase

from aizynthmodels.route_distances.utils.data import accumulate_stats
from aizynthmodels.utils.scores import ScoreCollection
from aizynthmodels.utils.type_utils import StrDict


class _TreeLstmWithPreCompression(torch.nn.Module):
    def __init__(self, fp_size: int, lstm_size: int, dropout_prob: float) -> None:
        super().__init__()
        self._compression = torch.nn.Sequential(
            torch.nn.Linear(fp_size, lstm_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(lstm_size, lstm_size),
            torch.nn.ReLU(),
        )
        self._tree_lstm = TreeLSTMBase(lstm_size, lstm_size)

    def forward(self, tree_batch: StrDict) -> torch.Tensor:
        """
        Forward pass

        :param tree_batch: collated trees from the `route_distances.utils.collate_trees` function.
        :return: the LSTM representation of the first nodes
        """
        features = self._compression(tree_batch["features"])
        lstm_output, _ = self._tree_lstm(
            features,
            tree_batch["node_order"],
            tree_batch["adjacency_list"],
            tree_batch["edge_order"],
        )
        # Only save value of top-node
        lstm_output = torch.stack([t[0, :] for t in torch.split(lstm_output, tree_batch["tree_sizes"], dim=0)])
        return lstm_output


class LstmDistanceModel(lightning.LightningModule):
    """
    Model for computing the distances between two synthesis routes

    :param fp_size: the length of the fingerprint vector
    :param lstm_size: the size o the LSTM cell
    :param dropout_prob: the dropout probability
    :param learning_rate: the initial learning rate of the optimizer
    :param weight_decay: weight decay factor of the optimizer
    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore="config")

        if config:
            for key in config:
                self.save_hyperparameters({key: config[key]})

        logging.info("Model hyperparameters:")
        logging.info(self.hparams)

        self._tree_lstm = _TreeLstmWithPreCompression(
            self.hparams.fp_size,
            self.hparams.lstm_size,
            self.hparams.dropout_prob,
        )
        self._pdist = torch.nn.PairwiseDistance(p=2)
        self._loss_func = torch.nn.MSELoss()

        self._lr = self.hparams.learning_rate
        self._weight_decay = self.hparams.weight_decay

        self.scores: Optional[ScoreCollection] = None

        self.validation_step_outputs: List[StrDict] = []
        self.test_step_outputs: List[StrDict] = []

    def forward(self, tree_data: StrDict) -> torch.Tensor:
        """
        Calculate the pairwise distances between the input trees

        :param tree_data: collated trees from the `route_distances.utils.collate_trees` function.
        :return: the distances in condensed form
        """
        lstm_enc = self._tree_lstm(tree_data)
        return torch.pdist(lstm_enc)

    def training_step(self, batch: StrDict, _) -> torch.Tensor:
        """
        One step in the training loop

        :param batch: collated pair data from the `route_distances.utils.collate_batch` function
        :param _: ignored
        :return: the loss tensor
        """
        pred = self._calculate_distance(batch)
        loss = self._loss_func(pred, batch["ted"])
        if self.scores:
            scores = self.scores.apply(pred, batch["ted"])
            for key, val in scores.items():
                self.log(f"train_{key}_step", val, prog_bar=True)
        self.log("train_loss_step", loss.item())
        return loss

    def validation_step(self, batch: StrDict, _) -> StrDict:
        """
        One step in the validation loop

        :param batch: collated pair data from the `route_distances.utils.collate_batch` function
        :param _: ignored
        :return: the validation metrics
        """
        metrics, _ = self._val_and_test_step(batch, "val")
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self) -> None:
        """Log the average validation metrics"""
        outputs = self.validation_step_outputs
        self._log_average_metrics(outputs)
        self.validation_step_outputs = []  # free memory

    def test_step(self, batch: StrDict, _) -> StrDict:
        """
        One step in the test loop

        :param batch: collated pair data from the `route_distances.utils.collate_batch` function
        :param _: ignored
        :return: the test metrics
        """
        metrics, predictions = self._val_and_test_step(batch, "test")
        metrics.update(predictions)
        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self) -> None:
        """Log the average test metrics"""
        outputs = self.test_step_outputs
        self._log_average_metrics(outputs)
        self.test_step_outputs = []  # free memory

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Adam], List[StrDict]]:
        """Setup the Adam optimiser and scheduler"""
        optim = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim),
            "monitor": "val_loss",
        }
        return [optim], [scheduler]

    def _calculate_distance(self, batch: StrDict) -> torch.Tensor:
        lstm_out1 = self._tree_lstm(batch["tree1"])
        lstm_out2 = self._tree_lstm(batch["tree2"])
        return self._pdist(lstm_out1, lstm_out2)

    def _log_average_metrics(self, outputs: List[StrDict]) -> None:
        accum = accumulate_stats(outputs)
        for key, value in accum.items():
            self.log(key, value / len(outputs))

    def _val_and_test_step(self, batch: StrDict, prefix: str) -> StrDict:
        self.eval()
        pred = self._calculate_distance(batch)
        loss = self._loss_func(pred, batch["ted"])

        metrics = {
            f"{prefix}_loss": loss.item(),
        }

        if self.scores:
            scores = self.scores.apply(pred, batch["ted"])
            keys = list(scores.keys())
            for key in keys:
                scores[f"{prefix}_{key}"] = scores.pop(key)
            metrics.update(scores)

        predictions = {
            "predictions": pred.cpu().detach().numpy(),
            "ground_truth": batch["ted"].cpu().detach().numpy(),
        }
        return metrics, predictions
