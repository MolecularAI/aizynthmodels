from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple
    from aizynthmodels.utils.scores import ScoreCollection
    from aizynthmodels.chemformer.sampler import SMILESSampler
    from aizynthmodels.utils.type_utils import StrDict
    from omegaconf import DictConfig
    from torch import Tensor

from aizynthmodels.chemformer.utils.models import (
    basic_transformer_encode,
    calculate_token_accuracy,
    positional_embeddings,
)

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------- Base Transformer Model -----------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BaseTransformer(pl.LightningModule):
    """
    Base transformer model.

    :ivar config: The model hyperparameters config.
    :ivar embedding: The embedding of the neural network.
    :ivar dropout: The dropout layer of the neural network.
    :ivar encoder: An instance of the encoder layers.

    :param config: The model hyperparameters config. This contains the following keys -
        - d_model: int
        - num_layers: int
        - pad_token_idx: int
        - dim_feedforward: int
        - max_seq_len: int
        - activation: str
        - total_steps: int
        - warm_up_steps: int
        - dropout: float
        - n_predictions: int
        - batch_first: bool
        - vocabulary_size: int
        - optimizer: dict with keys - scheduler, learning_rate, weight_decay, betas
        - batch_first: whether to use tensor format [batch, seq] (True) or [seq, batch] (False)
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

        if "batch_first" not in self.hparams:
            self.save_hyperparameters({"batch_first": False})

        if "d_model" in self.hparams:
            if "optimizer" in self.hparams:
                self.optimizer = self.hparams.optimizer
            else:
                # For loading old Chemformer version checkpoints
                self.optimizer: StrDict = {
                    "learning_rate": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                    "scheduler": self.hparams.schedule,
                    "betas": self.hparams.get("betas", [0.9, 0.999]),
                }

            self.emb = nn.Embedding(
                self.hparams.vocabulary_size,
                self.hparams.d_model,
                padding_idx=self.hparams.pad_token_idx,
            )
            self.register_buffer(
                "pos_emb",
                positional_embeddings(self.hparams.d_model, self.hparams.max_seq_len),
            )

            self.dropout = nn.Dropout(self.hparams.dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                self.hparams.d_model,
                self.hparams.num_heads,
                self.hparams.d_feedforward,
                self.hparams.dropout,
                self.hparams.activation,
                batch_first=self.hparams.batch_first,
                norm_first=True,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                self.hparams.num_layers,
                norm=nn.LayerNorm(self.hparams.d_model),
            )

        self.sampler: Optional[SMILESSampler] = None
        self.scores: Optional[ScoreCollection] = None

        self.validation_step_outputs: List[StrDict] = []
        self.test_step_outputs: List[StrDict] = []

    def _init_params(self) -> None:
        """Apply Xavier uniform initialisation of learnable weights."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, batch: StrDict) -> Tensor:
        """Construct the memory embedding for an encoder input.
        :param batch: A dictionary of the model inputs. This should contain keys:
            - "encoder_input": tensor of token_ids of shape (src_len, batch_size)
            - "encoder_pad_mask": bool tensor of padded elements of shape (src_len, batch_size)
        :return: A tensor of the encoder memory of shape (seq_len, batch_size, d_model)
                if not self.hparams.batch_first, else shape (batch_size, seq_len, d_model).
        """
        encoder_input = batch["encoder_input"].transpose(0, 1) if self.hparams.batch_first else batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        return basic_transformer_encode(self, encoder_input, encoder_pad_mask)

    def forward(self, batch: StrDict) -> StrDict:
        """Apply the input to the model. Should be implemented by sub-class.

        :param batch: A dictionary of the model inputs.
        :return: A dictionary of the model output.
        """
        raise NotImplementedError()

    def loss(self, batch_input: StrDict, model_output: StrDict) -> Tensor:
        """Calculate the loss for the model. Should be implemented by sub-class.

        :param batch_input: Input given to model.
        :param model_output: Output from the model.
        :return: The loss as a singleton tensor.
        """
        raise NotImplementedError()

    def sample_predictions(self, batch: StrDict, **kwargs: Any) -> Tuple[List[str], List[float]]:
        """Sample predictions from the model. Should be implemented by sub-class.

        :param batch: Input given to model.
        :return: Tuple of predictions and score (such as log lhs or probabilities)
        """
        raise NotImplementedError()

    def training_step(self, batch: StrDict, batch_idx: int) -> Tensor:
        self.train()
        model_output = self.forward(batch)
        loss = self.loss(batch, model_output)
        self.log("training_loss", loss, on_step=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: StrDict, batch_idx: int) -> StrDict:
        step_outputs = self._val_and_test_step(batch)
        metrics = {
            f"validation_{key}": step_outputs[key].item()
            for key in ["loss", "token_accuracy"]
            if step_outputs.get(key) is not None
        }
        if step_outputs.get("sampled_metrics"):
            metrics.update(step_outputs["sampled_metrics"])
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_outputs = self._avg_dicts(outputs)
        self.log_dict(avg_outputs)
        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict[str, List[Any]], batch_idx: int) -> StrDict:
        step_outputs = self._val_and_test_step(batch)
        metrics = {
            "batch_idx": batch_idx,
            "test_loss": step_outputs["loss"].item(),
            "test_token_accuracy": step_outputs["token_accuracy"],
            "log_lhs": step_outputs["log_likelihoods"],
            "sampled_molecules": step_outputs["sampled_smiles"],
            "target_smiles": step_outputs["target_smiles"],
            "sampling_time": step_outputs["sampling_time"],
        }
        metrics.update(step_outputs["sampled_metrics"])

        if self.sampler.sample_unique:
            metrics_unique = self._compute_scores_for_unique_smiles(batch["target_smiles"])
            metrics.update(metrics_unique)
        self.test_step_outputs.append(metrics)
        return metrics

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer["learning_rate"],
            weight_decay=self.optimizer["weight_decay"],
            betas=tuple(self.optimizer["betas"]),
        )

        logging.info(f"Using {self.optimizer['scheduler']} LR schedule.")
        scheduler = self._setup_scheduler()
        return [self.optim], [scheduler]

    def construct_lambda_lr(self, step):  # TODO
        if self.optimizer["scheduler"] == "transformer":
            assert (
                self.hparams.warm_up_steps is not None
            ), "A value for warm_up_steps is required for transformer LR schedule"
            multiplier = self.hparams.d_model**-0.5
            step = 1 if step == 0 else step  # Stop div by zero errors
            lr = min(step**-0.5, step * (self.hparams.warm_up_steps**-1.5))
            learning_rate = self.optimizer["learning_rate"] * multiplier * lr
        else:
            if self.hparams.warm_up_steps is not None and step < self.hparams.warm_up_steps:
                learning_rate = (self.optimizer["learning_rate"] / self.hparams.warm_up_steps) * step
        return learning_rate

    def _compute_scores_for_unique_smiles(self, target_smiles: List[str]) -> StrDict:
        metrics = {}
        sampled_smiles_unique = self.sampler.smiles_unique
        log_lhs_unique = self.sampler.log_lhs_unique
        metrics = {
            "log_lhs(unique)": log_lhs_unique,
            "sampled_molecules(unique)": sampled_smiles_unique,
        }
        scores_unique = self.scores.apply(sampled_smiles_unique, target_smiles)

        drop_cols = ["fraction_invalid", "fraction_unique", "top1_tanimoto_similarity"]
        metrics.update({f"{key}(unique)": val for key, val in scores_unique.items() if key not in drop_cols})
        return metrics

    def _setup_scheduler(self) -> StrDict:
        if self.optimizer["scheduler"] == "cycle":
            scheduler = OneCycleLR(
                self.optim,
                self.optimizer["learning_rate"],
                total_steps=self.hparams.num_steps,
            )
        else:
            scheduler = FuncLR(self.optim, lr_lambda=self.construct_lambda_lr)

        return {"scheduler": scheduler, "interval": "step"}

    def _val_and_test_step(self, batch: StrDict) -> StrDict:
        self.eval()
        step_outputs = defaultdict(list)
        with torch.no_grad():
            model_output = self.forward(batch)

            start_time = time.time()
            sampled_smiles, log_likelihoods = self.sample_predictions(batch)
            sampling_time = time.time() - start_time

        sampled_metrics = {}
        if len(sampled_smiles) > 0:
            sampled_metrics = self.scores.apply(sampled_smiles, batch["target_smiles"])

        step_outputs = {
            "target_smiles": batch["target_smiles"],
            "loss": self.loss(batch, model_output),
            "token_accuracy": calculate_token_accuracy(batch, model_output),
            "sampled_smiles": sampled_smiles,
            "log_likelihoods": log_likelihoods,
            "sampling_time": sampling_time,
            "sampled_metrics": sampled_metrics,
        }
        return step_outputs

    def _avg_dicts(self, colls):
        complete_dict = {k: [c[k] for c in colls] for k in colls[0].keys()}  # list[dict] -> dict[list]
        avg_dict = {k: sum(l) / len(l) for k, l in complete_dict.items()}
        return avg_dict


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]
