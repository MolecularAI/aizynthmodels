from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from rdkit import RDLogger

from aizynthmodels.chemformer.sampler.node import __name__ as sampler_module
from aizynthmodels.chemformer.utils.sampler import EOS, LogicalOr, MaxLength
from aizynthmodels.utils.loading import load_item
from aizynthmodels.utils.smiles import uniqueify_sampled_smiles

if TYPE_CHECKING:
    from typing import Any, Dict, Tuple

    from aizynthmodels.chemformer.models.base_transformer import BaseTransformer
    from aizynthmodels.chemformer.sampler.node import BaseSamplerNode
    from aizynthmodels.chemformer.utils.sampler import Criterion
    from aizynthmodels.utils.tokenizer import SMILESTokenizer
    from aizynthmodels.utils.type_utils import StrDict

RDLogger.DisableLog("rdApp.*")


class SMILESSampler:
    """
    Base sampler class for generating and scoring SMILES.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        max_sequence_length: int,
        device: str = "cuda",
        sample_unique: bool = True,
        sampler_node: str = "BeamSearchSampler",
        batch_size: int = 32,
    ) -> None:
        """
        :param tokenizer: Tokenizer with vocabulary.
        :param max_sequence_length: Maximum generated sequence length.
        :param device: "cuda" or "cpu".
        :param sampled_unique:  Whether to return unique beam search solutions.
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.sampling_strategy = sampler_node
        self.sample_unique = sample_unique

        self.smiles_unique = None
        self.log_lhs_unique = None

        node_kwargs = {"tokenizer": self.tokenizer, "device": self.device, "batch_size": batch_size}

        self.node = self._load_node_from_config(self.sampling_strategy, node_kwargs)
        return

    def _load_node_from_config(self, sampler_class: str, sampler_kwargs: StrDict) -> BaseSamplerNode:
        """
        Load a sampler node from a class and configuration dictionary

        If a sampler is not defined in the ``aizynthmodels.chemformer.utils.samplers`` module, the module
        name can be appended, e.g. ``mypackage.scoring.AwesomeSampler``.

        The values of the configuration is passed directly to the sampler
        class along with the ``config`` parameter.

        :param sampler_class: the sample class name to load.
        :param sampler_kwargs: the arguments passed to the BaseSamplerNode.
        :return: The sampler node (base algorithm, such as greedy, beam search,
            multinomial, etc.).
        """
        cls, kwargs, config_str = load_item(sampler_class, sampler_module, sampler_kwargs)
        obj = cls(**kwargs)
        logging.info(f"Loaded sampler: '{sampler_class}'{config_str}")
        return obj

    def sample_molecules(
        self,
        model: BaseTransformer,
        batch_input: Dict[str, Any],
        n_predictions: int,
        return_tokenized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample sequences of tokens from the model using the current sampling method.

        :param model: The transformer base model (e.g. BARTModel)
        :param batch_input: The input, X, to the network
        :param n_predictions: Number of predictions to return (e.g. beam_size in beam search)
        :param return_tokenized: whether to return the sampled tokens (True), or the
            converted SMILES (False). Defaults to False.

        :return: The sampled smiles (or token indices) and log-likelihoods
        """

        stop_criterion = LogicalOr((MaxLength(self.max_sequence_length - 1), EOS()))

        if self.device is None:
            self.device = next(model.parameters()).device
            self.node.device = self.device

        self.node.initialize(model, batch_input, n_predictions)
        logging.info(f"Sampling {n_predictions} predictions.")

        self._run_sampling(stop_criterion)

        Y = self.node.y.detach().cpu().numpy()
        log_lhs = (self.node.loglikelihood.detach().cpu().to(torch.float32).numpy()).reshape(-1, n_predictions)

        tokens = self.tokenizer.convert_ids_to_tokens(Y)

        if return_tokenized:
            return Y.tolist(), log_lhs.tolist()

        sampled_smiles = np.asarray(self.tokenizer.detokenize(tokens, truncate_at_end_token=True)).reshape(
            (-1, n_predictions)
        )

        sampled_smiles, log_lhs = self.node.sort(sampled_smiles, log_lhs)
        self._set_unique_samples(sampled_smiles, log_lhs, n_predictions)

        return sampled_smiles.tolist(), log_lhs.tolist()

    def _set_unique_samples(self, sampled_smiles: np.ndarray, log_lhs: np.ndarray, max_n_unique: int) -> None:
        """Deduplicate the sampled SMILES and keep the first max_n_unique (unique) samples."""
        if self.sample_unique:
            n_predictions = sampled_smiles.shape[1]
            if n_predictions == 1:
                self.smiles_unique = sampled_smiles
                self.log_lhs_unique = log_lhs
            else:
                (
                    self.smiles_unique,
                    self.log_lhs_unique,
                ) = uniqueify_sampled_smiles(sampled_smiles, log_lhs, max_n_unique)

    def _run_sampling(self, stop_criterion: Criterion) -> BaseSamplerNode:
        """
        Executing sampling using the current sampler node.

        :param node: The sampler node that computes the log-likelihoods and
        """

        while not stop_criterion(self.node):
            a = self.node.get_actions()
            self.node.action(a)

        a = self.node.get_actions()

        end_tokens = self.node.tokenizer["end"] * torch.logical_not(self.node.ll_mask).type(self.node.y.dtype)
        self.node.y = torch.cat((self.node.y, end_tokens.view(-1, 1)), dim=-1)
        ll_tail = a[torch.arange(len(a)), end_tokens] * torch.logical_not(self.node.ll_mask).type(a.dtype)
        self.node.loglikelihood = self.node.loglikelihood + ll_tail.view(-1, 1)
        return self.node
