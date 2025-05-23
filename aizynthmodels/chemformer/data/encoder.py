""" Module containing the BatchEncoder """

import logging
from typing import Any, List, Optional, Tuple

import torch

from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer, ListOfStrList, TokensMasker


class BatchEncoder:
    """
    Encodes a sequence for the Chemformer model

    This procedure includes:
        1. Tokenization
        2. Optional masking
        3. Padding
        4. Optional adding separation token to the end
        5. Checking of sequence lengths and possibly truncation
        6. Conversion to pytorch.Tensor

    Encoding is carried out by

    .. code-block::

        id_tensor, mask_tensor = encoder(batch, mask=True)

    where `batch` is a list of strings to be encoded and `mask` is
    a flag that can be used to toggled the masking.

    :param tokenizer: the tokenizer to use
    :param masker: the masker to use
    :param max_seq_len: the maximum allowed list length
    """

    def __init__(
        self,
        tokenizer: ChemformerTokenizer,
        masker: Optional[TokensMasker],
        max_seq_len: int,
    ):
        self._logger = logging.Logger("batch-encoder")
        self._tokenizer = tokenizer
        self._masker = masker
        self._max_seq_len = max_seq_len

    def __call__(
        self,
        batch: List[str],
        mask: bool = False,
        add_sep_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenizer.tokenize(batch)
        if mask and self._masker is not None:
            tokens, _ = self._masker(tokens)
        tokens, pad_mask = self._pad_seqs(tokens, self._tokenizer.special_tokens["pad"])

        if add_sep_token:
            sep_token = self._tokenizer.special_tokens["sep"]
            tokens = [itokens + [sep_token] for itokens in tokens]
            pad_mask = [imasks + [0] for imasks in pad_mask]

        tokens, pad_mask = self._check_seq_len(tokens, pad_mask)
        id_data = self._tokenizer.convert_tokens_to_ids(tokens)

        id_tensor = torch.stack(id_data)
        mask_tensor = torch.tensor(pad_mask, dtype=torch.bool)
        return id_tensor.transpose(0, 1), mask_tensor.transpose(0, 1)

    def _check_seq_len(self, tokens: ListOfStrList, mask: List[List[int]]) -> Tuple[ListOfStrList, List[List[int]]]:
        """Warn user and shorten sequence if the tokens are too long, otherwise return original"""

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self._max_seq_len:
            self._logger.warn(f"Sequence length {seq_len} is larger than maximum sequence size")

            tokens_short = [ts[: self._max_seq_len] for ts in tokens]
            mask_short = [ms[: self._max_seq_len] for ms in mask]

            return tokens_short, mask_short

        return tokens, mask

    @staticmethod
    def _pad_seqs(seqs: List[Any], pad_token: Any) -> Tuple[List[Any], List[int]]:
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks
