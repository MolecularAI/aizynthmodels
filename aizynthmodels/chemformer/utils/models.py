from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from typing import Optional

    from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
    from aizynthmodels.chemformer.models.base_transformer import BaseTransformer
    from aizynthmodels.utils.type_utils import StrDict


def set_model_hyperparams(
    config: DictConfig, tokenizer: ChemformerTokenizer, train_steps: int, n_classes: Optional[int] = None
) -> StrDict:
    model_hyperparams = OmegaConf.to_container(config.model_hyperparams, resolve=True)
    if config.mode.startswith("train"):
        total_steps = train_steps + 1
    else:
        total_steps = 0

    resume = config.get("resume", False) if config.get("model_path") else False

    if config.mode.startswith("train") and resume:
        model_hyperparams = {
            "num_steps": total_steps,
            "pad_token_idx": tokenizer["pad"],
            "vocabulary_size": len(tokenizer),
            "batch_first": model_hyperparams["batch_first"],
        }
        logging.info("Resuming training.")
    elif config.mode.startswith("train") or not config.get("model_path"):
        model_hyperparams["num_steps"] = total_steps
        model_hyperparams["pad_token_idx"] = tokenizer["pad"]
        model_hyperparams["vocabulary_size"] = len(tokenizer)
        model_hyperparams["task"] = config.task

    if n_classes:
        model_hyperparams["num_classes"] = n_classes
    return model_hyperparams


def calculate_token_accuracy(batch_input: StrDict, model_output: StrDict) -> float:
    """
    Calculates the token accuracy.

    :param batch_input: A dictionary of the model inputs.
    :param model_output: A dictionary of the model outputs.

    :return: The token accuracy.
    """
    token_ids = batch_input["target"]
    target_mask = batch_input["target_mask"]
    token_output = model_output["token_output"]

    target_mask = ~(target_mask > 0)
    _, pred_ids = torch.max(token_output.float(), dim=2)
    correct_ids = torch.eq(token_ids, pred_ids)
    correct_ids = correct_ids * target_mask

    num_correct = correct_ids.sum().float()
    total = target_mask.sum().float()

    accuracy = num_correct / total
    return accuracy


def construct_embeddings(
    token_ids: torch.Tensor,
    d_model: int,
    batch_first: bool,
    embedding: nn.Embedding,
    positional_embeddings: torch.Tensor,
):
    seq_len = token_ids.shape[1 if batch_first else 0]
    token_embeddings = embedding(token_ids)

    # Scaling the embeddings
    token_embeddings = token_embeddings * math.sqrt(d_model)
    positional_emb = positional_embeddings[:seq_len, :].unsqueeze(0)
    positional_emb = positional_emb.transpose(0, 1) if not batch_first else positional_emb
    embeddings = token_embeddings + positional_emb
    return embeddings


def generate_square_subsequent_mask(mask_size: int, device: str = "cpu") -> torch.Tensor:
    """
    Method copied from Pytorch nn.Transformer.
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).

    :param mask_size: Size of mask to generate.
    :return: Square autoregressive mask for decode.
    """
    mask = (torch.triu(torch.ones((mask_size, mask_size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def positional_embeddings(d_model: int, max_seq_len: int) -> torch.Tensor:
    """Produces a tensor of positional embeddings for the model.
    Returns a tensor of shape (max_seq_len, d_model) filled with positional embeddings,
    which are created from sine and cosine waves of varying wavelength.
    """
    encs = torch.tensor([dim / d_model for dim in range(0, d_model, 2)])
    encs = 10000**encs
    encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(max_seq_len)]
    encs = [torch.stack(enc, dim=1).flatten()[:d_model] for enc in encs]
    encs = torch.stack(encs)
    return encs


def basic_transformer_encode(
    model: BaseTransformer, encoder_input: torch.Tensor, encoder_pad_mask: torch.Tensor
) -> torch.Tensor:
    """Construct the memory embedding for an encoder input.

    :param encoder_input: tensor of token_ids of shape (src_len, batch_size)
    :param encoder_pad_mask: bool tensor of padded elements of shape (src_len, batch_size)
    :return: A tensor of the encoder memory of shape (seq_len, batch_size, d_model).
    """
    encoder_embeddings = model.dropout(
        construct_embeddings(
            encoder_input,
            model.hparams.d_model,
            model.hparams.batch_first,
            model.emb,
            model.pos_emb,
        )
    )
    memory = model.encoder(encoder_embeddings, src_key_padding_mask=encoder_pad_mask)
    return memory
