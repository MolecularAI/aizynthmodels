from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from aizynthmodels.chemformer.models.base_transformer import BaseTransformer
from aizynthmodels.chemformer.utils.models import basic_transformer_encode

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple, Union

    from omegaconf import DictConfig
    from torch import Tensor

    from aizynthmodels.utils.type_utils import StrDict


class TransformerClassifier(BaseTransformer):
    """
    Transformer classifier class.

    :param config: Configuration of the model hyperparameters.

    :ivar batch_first: Enforcing that batch-first encoding is used.
    :ivar encoder: An instance of the encoder layers.
    :ivar classifier: An instance of the mlp classifier layers.
    :ivar loss_function: The cross entropy loss function.
    :ivar softmax: Softmax of classifier outputs.
    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            config,
            **kwargs,
        )

        if len(self.hparams.num_hidden_nodes) > 0:
            layers = self.mlp_layer_stack(
                self.hparams.d_model, self.hparams.num_hidden_nodes[0], self.hparams.activation
            )
            for layer_idx in range(1, len(self.hparams.num_hidden_nodes)):
                layers.extend(
                    self.mlp_layer_stack(
                        self.hparams.num_hidden_nodes[layer_idx - 1],
                        self.hparams.num_hidden_nodes[layer_idx],
                        self.hparams.activation,
                    )
                )
            layers.append(nn.Linear(self.hparams.num_hidden_nodes[-1], self.hparams.num_classes))
        else:
            layers = [nn.Linear(self.hparams.d_model, self.hparams.num_classes)]

        self.classifier = nn.Sequential(*layers)
        self.loss_function = nn.CrossEntropyLoss(reduction="mean")
        self.softmax = nn.Softmax(dim=-1)
        self._init_params()

    def mlp_layer_stack(self, num_input_nodes: int, num_output_nodes: int, activation: str) -> List[Any]:
        """
        Create a stack of MLP layers (LayerNorm, Linear, activation function, and dropout)
        given the size of the current linear layer and type of activation function.

        :param num_input_nodes: number of input nodes in the linear layer
        :param num_output_nodes: number of output nodes in the linear layer
        :param activation: The activation function to use ('relu', 'gelu' or 'linear')
        :return: List of layers (stack) for the current parameter input.
        """

        if activation not in ["relu", "gelu", "linear"]:
            raise ValueError(f"'activation' can be either 'relu', 'gelu' or 'linear', got {activation}")

        layers = []
        layers.append(nn.LayerNorm(num_input_nodes))
        layers.append(nn.Linear(num_input_nodes, num_output_nodes))
        if activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "relu":
            layers.append(nn.RELU())

        layers.append(nn.Dropout(self.hparams.dropout))
        return layers

    def encode(self, batch: StrDict) -> StrDict:
        """Construct the memory embedding for an encoder input.
        :param batch: A dictionary of the model inputs. This should contain keys:
            - "encoder_input": tensor of token_ids of shape (src_len, batch_size)
            - "encoder_pad_mask": bool tensor of padded elements of shape (src_len, batch_size)
        :return: A tensor of the encoder memory of shape (batch_size, d_model).
        """
        encoder_input = batch["encoder_input"].transpose(0, 1) if self.hparams.batch_first else batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        memory = basic_transformer_encode(self, encoder_input, encoder_pad_mask)

        # Return the mean of the encoded sequences
        seq_dim = 1 if self.hparams.batch_first else 0

        n_tokens_per_sample = torch.sum(1 - encoder_pad_mask.float(), dim=seq_dim, keepdim=True)
        n_tokens_per_sample = torch.maximum(n_tokens_per_sample, torch.ones_like(n_tokens_per_sample))
        return torch.sum(memory, dim=seq_dim) / n_tokens_per_sample

    def forward(self, batch: StrDict) -> StrDict:
        """Apply SMILES strings to the model and predict class probabilities.
        :param batch: A dictionary of the model inputs. This should contain keys:
            - "encoder_input": tensor of token_ids of shape (src_len, batch_size)
            - "encoder_pad_mask": bool tensor of padded elements of shape (src_len, batch_size)
        :return: The predicted probabilities associated to each class.
        """
        memory = self.encode(batch)
        logits = self.classifier(memory)
        return {"logits": logits}

    def generator(self, logits: Tensor) -> Tensor:
        probabilities = self.softmax(logits)
        return probabilities

    def loss(self, batch: StrDict, model_output: StrDict) -> Tensor:
        """Calculate the loss for the model.
        :param batch: Input given to the model.
        :param model_output: Output from the model.
        :return: The loss as a singleton tensor.
        """
        loss = self.loss_function(model_output["logits"], batch["class_indicator"])
        return loss

    def sample_predictions(
        self, batch: StrDict, **kwargs: Any
    ) -> Tuple[Union[List[Union[str, float]], np.ndarray], ...]:
        """Rank classes by predicted class probabilities.
        :param batch: Input given to the model.
        :return: A tuple of molecule SMILES strings and log lhs.
        """

        model_outputs = self.forward(batch)
        probabilites_all_classes = self.generator(model_outputs["logits"])

        probabilities, predicted_labels = torch.topk(probabilites_all_classes, self.n_predictions, dim=1)

        predicted_labels = predicted_labels.detach().cpu().numpy()
        probabilities = probabilities.detach().cpu().numpy()
        return predicted_labels, probabilities

    def test_step(self, batch: StrDict, batch_idx: int) -> StrDict:
        step_outputs = self._val_and_test_step(batch)
        metrics = {
            "batch_idx": batch_idx,
            "test_loss": step_outputs["loss"].item(),
            "probabilities": step_outputs["probabilities"],
            "predictions": step_outputs["predictions"],
            "ground_truth": batch["label"],
            "sampling_time": step_outputs["t_sampling"],
        }

        metrics.update(step_outputs["sampled_metrics"])
        self.test_step_outputs.append(metrics)
        return metrics

    def _val_and_test_step(self, batch: StrDict) -> StrDict:
        self.eval()
        step_outputs = defaultdict(list)
        with torch.no_grad():
            model_output = self.forward(batch)

            t0 = time.time()
            predictions, probabilities = self.sample_predictions(batch)
            sampling_time = time.time() - t0

        if self.scores:
            sampled_metrics = self.scores.apply(predictions, batch["label"])
        else:
            sampled_metrics = {}

        step_outputs = {
            "target_smiles": batch.get("target_smiles"),
            "loss": self.loss(batch, model_output),
            "predictions": predictions,
            "probabilities": probabilities,
            "t_sampling": sampling_time,
            "sampled_metrics": sampled_metrics,
        }
        return step_outputs
