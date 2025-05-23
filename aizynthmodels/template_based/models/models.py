""" Module containing the model basis for template-based retrosynthesis
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from rxnutils.chem.template import ReactionTemplate
from torch import nn

from aizynthmodels.utils.type_utils import StrDict


class MulticlassClassifier(pl.LightningModule):
    """
    The multiclass-classifier model that is the basis for the
    template-based retrosynthesis model.

    :param num_features: the number of input features
    :param num_classes: the number of output classes
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

        layers = []
        layers.append(nn.Linear(self.hparams.num_features, self.hparams.num_hidden_nodes))
        for layer_idx in range(self.hparams.num_hidden_layers):
            if layer_idx > 0:
                layers.append(nn.Linear(self.hparams.num_hidden_nodes, self.hparams.num_hidden_nodes))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(self.hparams.dropout))
        layers.append(nn.Linear(self.hparams.num_hidden_nodes, self.hparams.num_classes))

        self._model = nn.Sequential(*layers)
        self._loss_func = nn.CrossEntropyLoss()

        self._learning_rate = self.hparams.learning_rate
        self._weight_decay = self.hparams.weight_decay
        self._templates = None
        self.scores = None
        self.n_predictions = None

        self._test_step_outputs: List[StrDict] = []

    def set_templates(self, template_filename: str) -> None:
        self._templates = pd.read_csv(template_filename, sep="\t", index_col=0)

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        """
        Computes the probabilities of each of the classes for the input vector

        :param feature_vector: the input vector
        """
        return torch.softmax(self._model(feature_vector), -1)

    def sample_predictions(self, batch: Dict[str, torch.Tensor], n_predictions: int) -> StrDict:
        """
        Predicts and ranks templates according to their probabilities.
        If templates have been set in the class and the batch contains the input product SMILES,
        the (ranked) templates will be applied to the product to generate predicted
        reactant SMILES.

        :param batch: the data of the current batch (contains input and ground-truth).
        :param n_predictions: the number of predicted templates to return.
        :returns: A dictionary containing the predictions (templates or SMILES), probabilities
            and the ground-truth.
        """
        template_probabilities = self.forward(batch["input"])
        label_probabilities, predicted_labels = torch.topk(template_probabilities, n_predictions, axis=1)

        if self._templates is None or batch.get("product") is None:
            # Return template predictions
            return {
                "predictions": predicted_labels.cpu().detach().numpy().tolist(),
                "probabilities": label_probabilities.cpu().detach().numpy().tolist(),
                "ground_truth": batch["label"].argmax(-1).cpu().detach().numpy().tolist(),
            }

        # Apply templates of the top-n predictions to products
        predicted_smiles, smiles_probabilities = self._labels_to_smiles(
            batch, predicted_labels.detach().cpu().numpy(), label_probabilities.detach().cpu().numpy()
        )
        return {
            "predictions": predicted_smiles,
            "probabilities": smiles_probabilities,
            "ground_truth": batch["reactant"],
        }

    def _labels_to_smiles(
        self, batch: StrDict, predicted_labels: np.ndarray, probabilities: np.ndarray
    ) -> Tuple[List[str], List[str]]:
        """
        Apply templates to the input product to obtain the predicted SMILES.
        The predicted SMILES are grouped according to their corresponding template rank.
        """
        predicted_smiles = []
        smiles_probabilities = []
        for idxs, top_k_probabilities, product in zip(predicted_labels, probabilities, batch["product"]):
            top_k_smiles = []
            top_k_smiles_prob = []
            for smarts, probability in zip(self._templates.loc[idxs, "retro_template"].to_list(), top_k_probabilities):
                template_reaction = ReactionTemplate(smarts=smarts, direction="retro")
                predicted_reactants = template_reaction.apply(product)

                top_k_smiles.append([])
                top_k_smiles_prob.append(probability)

                for reactants in predicted_reactants:
                    top_k_smiles[-1].append(".".join(reactants))

                if len(predicted_reactants) == 0:
                    # if the template could not be applied, append a dummy SMILES
                    top_k_smiles[-1].append("dummy-smiles")

            predicted_smiles.append(top_k_smiles)
            smiles_probabilities.append(top_k_smiles_prob)

        return predicted_smiles, smiles_probabilities

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The lightning training step"""
        logits = self._model(batch["input"])
        loss = self._loss_func(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: StrDict, _) -> StrDict:
        """The lightning validation step"""
        with torch.no_grad():
            logits = self._model(batch["input"])
            loss = self._loss_func(logits, batch["label"])
            _, predicted_labels = torch.topk(logits, 50, axis=1)

        metrics = self.scores.apply(predicted_labels, batch["label"].argmax(-1))
        self.log("val_loss", loss, on_epoch=True)
        self._log_metrics(metrics, "val")
        return metrics

    def test_step(self, batch: StrDict, _) -> None:
        """The lightning test step: generating and scoring predictions."""
        with torch.no_grad():
            predictions = self.sample_predictions(batch, self.n_predictions)

        kwargs = {}
        if "product" not in batch or self._templates is None:
            kwargs = {"is_canonical": True}

        metrics = self.scores.apply(predictions["predictions"], predictions["ground_truth"], **kwargs)
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

    def load_weights_from_onnx(self, filename: str) -> None:
        """Load weights from a trained ONNX model, e.g.
        a converted Tensorflow model
        """
        onnx_model = onnx.load(filename)
        graph = onnx_model.graph
        logging.info("Loading the following layers with sizes from ONNX model: ")
        onnx_weights = dict()
        for init in graph.initializer:
            onnx_weights[init.name] = onnx.numpy_helper.to_array(init)
        self.update_weights(onnx_weights)

    def update_weights(self, weights: Dict[str, np.ndarray], raise_exception: bool = True) -> None:
        """
        Updates the weights from a collection of weights, e.g.
        from another model. The matching of layers are based on
        matrix shapes.

        :param weights: a dictionary of weights
        :param raise_exception: if True and no matching is possible, raise an exception
        """
        for name, param in self._model.named_parameters():
            found = None
            for other_name, mat in weights.items():
                mat_conv = torch.from_numpy(mat.copy().T)
                if mat_conv.shape == param.shape:
                    found = other_name
                    param.data = mat_conv.data
                    break
            if found is None and raise_exception:
                raise ValueError(f"Could not find matching matrix for: {name}. Model left in corrupt state")
            logging.info(f"Set weights of {name} to weights of {found}")

    def _log_metrics(self, metrics: StrDict, prefix: str) -> StrDict:
        for key, val in metrics.items():
            self.log(f"{prefix}_{key}", val)
        return
