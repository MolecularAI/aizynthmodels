""" Module containing the model basis for template-based retrosynthesis
"""

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from aizynthmodels.utils.type_utils import StrDict


class ClassificationModel(pl.LightningModule):
    """
    The classification model that is the basis for the
    quick-filter model.

    :param config: the optional omega config containing all hyperparameters
    :param num_features: the number of input features
    :param num_hidden_layers: the number of hidden layers
    :param num_hidden_nodes: the size of the hidden layers
    :param dropout: the probability of the dropout layer
    :param learning_rate: the learning rate of the optimizer
    :param weight_decay: the weight decay parameter of the optimizer
    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="config")

        if config:
            for key in config:
                self.save_hyperparameters({key: config[key]})

        _input_encoders = []
        for _ in range(2):
            layers = []
            layers.append(nn.Linear(self.hparams.num_features, self.hparams.num_hidden_nodes))
            for layer_idx in range(self.hparams.num_hidden_layers):
                if layer_idx > 0:
                    layers.append(nn.Linear(self.hparams.num_hidden_nodes, self.hparams.num_hidden_nodes))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(self.hparams.dropout))
            _input_encoders.append(nn.Sequential(*layers))
        self._product_encoder, self._rxn_encoder = _input_encoders
        self._similarity = nn.CosineSimilarity()
        self._loss_func = nn.BCEWithLogitsLoss()

        self._learning_rate = self.hparams.learning_rate
        self._weight_decay = self.hparams.weight_decay
        self.scores = None
        self.n_predictions = None

        self._test_step_outputs: List[StrDict] = []

    def forward(self, product_feature_vector: torch.Tensor, reaction_feature_vector: torch.Tensor) -> torch.Tensor:
        """
        Computes the probabilities of each of the classes for the input vector

        :param product_feature_vector: the input vector for the product
        :param reaction_feature_vector: the input vector for the reaction
        """
        return torch.sigmoid(self._calc_logits(product_feature_vector, reaction_feature_vector))

    def sample_predictions(self, batch: Dict[str, torch.Tensor], n_predictions: int) -> StrDict:
        """
        :param batch: the data of the current batch (contains input and ground-truth).
        :param n_predictions: the number of predicted templates to return.
        :returns: A dictionary containing the predictions (templates or SMILES), probabilities
            and the ground-truth.
        """
        probs = self.forward(batch["product_input"], batch["reaction_input"])
        predictions = (probs > self.hparams.threshold).float()
        return {
            "predictions": predictions.cpu().detach().numpy().tolist(),
            "probabilities": probs.cpu().detach().numpy().tolist(),
            "ground_truth": batch["label"].cpu().detach().numpy().tolist(),
        }

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The lightning training step"""
        logits = self._calc_logits(batch["product_input"], batch["reaction_input"])
        loss = self._loss_func(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: StrDict, _) -> None:
        """The lightning validation step"""
        with torch.no_grad():
            logits = self._calc_logits(batch["product_input"], batch["reaction_input"])
            loss = self._loss_func(logits, batch["label"])

        probs = torch.sigmoid(logits)
        predictions = (probs > self.hparams.threshold).float()
        metrics = self.scores.apply(predictions, batch["label"])
        self.log("val_loss", loss, on_epoch=True)
        self._log_metrics(metrics, "val")
        return metrics

    def test_step(self, batch: StrDict, _) -> None:
        """The lightning test step: generating and scoring predictions."""
        with torch.no_grad():
            predictions = self.sample_predictions(batch, self.n_predictions)

        metrics = self.scores.apply(torch.tensor(predictions["predictions"], device=self.device), batch["label"])
        metrics.update(predictions)
        self._test_step_outputs.append(metrics)
        return metrics

    def configure_optimizers(self) -> Tuple[List[torch.optim.Adam], List[StrDict]]:
        """The lightning setup of the optimizer"""
        optim = torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5, eps=1e-6),
            "monitor": "val_loss",
        }
        return [optim], [scheduler]

    def _calc_logits(self, product_feature_vector: torch.Tensor, reaction_feature_vector: torch.Tensor) -> torch.Tensor:
        enc1 = self._product_encoder(product_feature_vector)
        enc2 = self._rxn_encoder(reaction_feature_vector)
        return self._similarity(enc1, enc2)

    def _log_metrics(self, metrics: StrDict, prefix: str) -> StrDict:
        for key, val in metrics.items():
            self.log(f"{prefix}_{key}", val)
