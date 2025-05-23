from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from aizynthmodels.utils.type_utils import StrDict
    from typing import Any, List, Optional, Tuple, Union
    from torch import Tensor
    from omegaconf import DictConfig

from aizynthmodels.chemformer.models.base_transformer import BaseTransformer
from aizynthmodels.chemformer.utils.models import construct_embeddings, generate_square_subsequent_mask


class BARTModel(BaseTransformer):
    """
    BART Model class.

    :ivar d_model:  The number of expected features in the input.
    :ivar max_seq_len: The maximum length of the sequence.
    :ivar batch_first:  If True, then the input and output tensors are provided as (batch, seq, feature).
    :ivar encoder: An instance of the encoder layers.
    :ivar decoder: An instance of the decoder layers.
    :ivar loss_function: The cross entropy loss function.
    :ivar token_fc: A linear transform.

    :param config: The model hyperparameters config.
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

        decoder_layer = nn.TransformerDecoderLayer(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.d_feedforward,
            self.hparams.dropout,
            self.hparams.activation,
            batch_first=self.hparams.batch_first,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            self.hparams.num_layers,
            norm=nn.LayerNorm(self.hparams.d_model),
        )

        self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=self.hparams.pad_token_idx)

        self.token_fc = nn.Linear(self.hparams.d_model, self.hparams.vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, batch: StrDict) -> StrDict:
        """Apply SMILES strings to the model.

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        :param batch: A dictionary of the model inputs. This should contain keys:
            - "encoder_input": tensor of token_ids of shape (src_len, batch_size)
            - "encoder_pad_mask": bool tensor of padded elements of shape (src_len, batch_size)
            - "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
            - "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)

        :return: A dictionary of the model output containing keys "token_output" and "model_output".
        """

        batch["memory_input"] = self.encode(batch)
        batch["memory_pad_mask"] = batch["encoder_pad_mask"].clone()
        model_output = self.decode(batch)

        token_output = self.token_fc(model_output)
        output = {"model_output": model_output, "token_output": token_output}
        return output

    def decode(self, batch: StrDict) -> Tensor:
        """Construct an output from a given decoder input

        :param batch: A dictionary of the model inputs. This should contain keys:
            - "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
            - "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            - "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
            - "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)

        :return: A tensor of the decoder memory of shape (seq_len, batch_size, d_model).
        """

        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)
        decoder_input = (
            batch["decoder_input"] if not self.hparams.batch_first else batch["decoder_input"].transpose(0, 1)
        )
        decoder_pad_mask = batch.get("decoder_pad_mask")
        decoder_pad_mask = decoder_pad_mask.transpose(0, 1) if decoder_pad_mask is not None else None

        decoder_embeddings = self.dropout(
            construct_embeddings(
                decoder_input,
                self.hparams.d_model,
                self.hparams.batch_first,
                self.emb,
                self.pos_emb,
            )
        )

        seq_len = decoder_embeddings.shape[0 if not self.hparams.batch_first else 1]
        tgt_mask = generate_square_subsequent_mask(seq_len, device=decoder_embeddings.device)

        decoder_output = self.decoder(
            decoder_embeddings,
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )
        return decoder_output if not self.hparams.batch_first else decoder_output.transpose(0, 1)

    def generator(self, decoder_output: Tensor) -> Tensor:
        token_log_probabilities = self.log_softmax(self.token_fc(decoder_output))
        return token_log_probabilities

    def loss(self, batch_input: StrDict, model_output: StrDict) -> Tensor:
        """Calculate the loss for the model.

        :param batch: Input given to the model.
        :param model_output: Output from the model.

        :return: The loss as a singleton tensor.
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def sample_predictions(self, batch: StrDict, return_tokenized: bool = False) -> Tuple[List[Union[str, float]], ...]:
        """Sample predictions (in BART's case SMILES) from the model.

        :param batch: Input given to the model.
        :return: A tuple of molecule SMILES strings and log lhs.
        """
        if self.n_predictions == 0:
            return [], []

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()
        sampled_smiles, log_lhs = self.sampler.sample_molecules(
            self,
            batch,
            self.n_predictions,
            return_tokenized=return_tokenized,
        )

        # Must remember to unfreeze!
        self.unfreeze()

        return sampled_smiles, log_lhs

    def decode_batch(self, batch: StrDict, return_last: bool = True) -> Tensor:
        """Construct an output from a given decoder input.

        :param batch: A dictionary of the model inputs. This should contain keys:
            - "decoder_input": tensor of token_ids of shape (tgt_len, batch_size)
            - "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            - "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
            - "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)

        :return: A tensor of the log probabilities.
        """
        if not self.hparams.batch_first:
            batch["memory_input"] = batch["memory_input"].permute(1, 0, 2)
        batch["decoder_input"] = batch["decoder_input"].transpose(0, 1)
        batch["memory_pad_mask"] = batch["memory_pad_mask"].transpose(0, 1)

        decoder_output = self.decode(batch)
        token_log_probabilities = self.generator(decoder_output)

        if return_last:
            return token_log_probabilities[-1, :, :]
        else:
            return token_log_probabilities

    def _calc_mask_loss(self, token_output, target, target_mask) -> Tensor:
        """Calculate the loss for the token prediction task.

        :param token_output: The token output from transformer. It is a tensor of shape
            (seq_len, batch_size, vocabulary_size).
        :param target: Original (unmasked) SMILES token ids from the tokenizer. It is a
            tensor of shape (seq_len, batch_size).
        :param target_mask: Pad mask for target tokens. It is a tensor of shape
            (seq_len, batch_size).

        :return: A singleton tensor of the loss computed using cross-entropy,
        """

        dim1, dim2 = tuple(target.size())

        token_pred = token_output.reshape((dim1 * dim2, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((dim1, dim2))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss
