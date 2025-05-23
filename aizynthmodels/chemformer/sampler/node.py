from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data as tud

if TYPE_CHECKING:
    from typing import Dict, Tuple, Union

    from aizynthmodels.chemformer.models.base_transformer import BaseTransformer
    from aizynthmodels.utils.tokenizer import SMILESTokenizer


class BaseSamplerNode:

    def __init__(self, tokenizer: SMILESTokenizer, device: Union[torch.device, str], batch_size: int) -> None:
        """
        Initialize a Node used for autoregression
        predictions, such as greedy search, multinomial
        sampling, or beam search.

        :param model: any autoregressive model
        :param x: a torch tensor representing
            additional data to pass to
            the regression model
        :param tokenizer: a tokenizer object
        :param device: device where to place the model and data
        :param batch_size: internal batch size used for the beam search
        """
        assert isinstance(device, torch.device) or isinstance(device, str)

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def initialize(
        self,
        model: BaseTransformer,
        batch: Dict[str, torch.Tensor],
        n_predictions: int,
    ):

        self.model = model
        self.n_predictions = n_predictions
        source_mask = batch["encoder_pad_mask"]

        self.model = self.model.eval()

        if next(self.model.parameters()).device != self.device:
            self.model = self.model.to(self.device)

        if source_mask.device != self.device:
            source_mask = source_mask.to(self.device)

        with torch.no_grad():
            self.x = self.model.encode(batch).detach()

        if not self.model.hparams.batch_first:
            self.x = self.x.permute(1, 0, 2)
        self.x_mask = source_mask.detach().transpose(0, 1)

        self.y = torch.ones((self.x.shape[0], 1), dtype=torch.long) * self.tokenizer["start"]
        self.y = self.y.detach()

        if self.x.device != self.device:
            self.x = self.x.to(self.device)
        if self.x_mask.device != self.device:
            self.x_mask = self.x_mask.to(self.device)

        if self.y.device != self.device:
            self.y = self.y.to(self.device)

        self.ll_mask = torch.Tensor([False])
        self.pos = 0

    def get_actions(self) -> torch.Tensor:
        next_loglikelihood = []

        X = {"decoder_input": self.y, "memory_input": self.x, "memory_pad_mask": self.x_mask}

        with torch.no_grad():
            ll = self.model.decode_batch(X)

        next_loglikelihood.append(ll)
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()
        return next_loglikelihood

    @staticmethod
    def sort(sampled_smiles: np.ndarray, log_lhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort sampled smiles if necessary."""
        for idx in range(log_lhs.shape[0]):
            sorted_idx = np.argsort(log_lhs[idx, :])[::-1]
            log_lhs[idx, :] = log_lhs[idx, sorted_idx]
            sampled_smiles[idx, :] = sampled_smiles[idx, sorted_idx]
        return sampled_smiles, log_lhs


class BeamSearchSampler(BaseSamplerNode):

    sampler_name = "BeamSearchSampler"

    def __init__(self, tokenizer: SMILESTokenizer, device: Union[torch.device, str], batch_size: int):
        super().__init__(tokenizer, device, batch_size)

    def action(self, next_loglikelhihood):
        if self.pos == 0:
            self._init_action(next_loglikelhihood)
        else:
            tokenizer_size = len(self.tokenizer)
            # set loglikehihood to the maxium (0)
            # when observed an eos_token
            next_loglikelhihood[self.ll_mask, :] = (
                torch.minimum(self.loglikelihood.min(), next_loglikelhihood.min()) - 1.0
            )
            next_loglikelhihood[self.ll_mask, self.tokenizer["end"]] = 0.0
            # done

            ll = (self.loglikelihood + next_loglikelhihood).view(-1, self.n_predictions, tokenizer_size)
            ll, idx = self._get_topk(ll.flatten(start_dim=1))

            # tricky indexing
            next_chars = torch.remainder(idx, tokenizer_size).flatten().unsqueeze(-1)
            best_candidates = (idx / tokenizer_size).long()
            if best_candidates.device != self.device:
                best_candidates = best_candidates.to(self.device)
            # done

            y = self.y.view(-1, self.n_predictions, self.y.shape[-1])
            i = torch.arange(len(y)).unsqueeze(-1).repeat(1, self.n_predictions).flatten()
            j = best_candidates.flatten()
            self.y = y[i, j].view(-1, self.y.shape[-1])

            self.y = torch.cat((self.y, next_chars), dim=-1)
            self.loglikelihood = ll.view(-1, 1)

            # update ll mask
            self.ll_mask = torch.any(self.y == self.tokenizer["end"], dim=-1)
        self.pos = self.pos + 1

    def get_actions(self) -> torch.Tensor:
        batch_size = self.batch_size
        next_loglikelihood = []

        local_dataset = tud.TensorDataset(self.x, self.x_mask, self.y)
        local_loader = tud.DataLoader(local_dataset, batch_size=batch_size)

        # make sure that the local_loader
        # will be iterated over only once
        iterator = iter(local_loader)  # noqa: F841

        with torch.no_grad():
            for x, x_mask, y in local_loader:
                if x.device != self.device:
                    x = x.to(self.device)
                if x_mask.device != self.device:
                    x_mask = x_mask.to(self.device)
                if y.device != self.device:
                    y = y.to(self.device)

                X = {"decoder_input": y, "memory_input": x, "memory_pad_mask": x_mask}

                ll = self.model.decode_batch(X)
                next_loglikelihood.append(ll)
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()

        return next_loglikelihood

    @staticmethod
    def sort(sampled_smiles: np.ndarray, log_lhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return sampled_smiles, log_lhs

    def _init_action(self, loglikelihood: torch.Tensor) -> None:
        # Perform the first step
        loglikelihood, next_chars = self._get_topk(loglikelihood)

        self.loglikelihood = loglikelihood.view(-1, 1)
        next_chars = next_chars.view(-1, 1)

        self.y = self.y.view(len(self.y), 1, -1).repeat(1, self.n_predictions, 1).view(-1, 1)
        self.x = self.x[:, None].repeat(1, self.n_predictions, 1, 1).view((-1,) + tuple(self.x.shape[1:]))
        self.x_mask = self.x_mask[:, None].repeat(1, self.n_predictions, 1).view((-1,) + tuple(self.x_mask.shape[1:]))

        self.y = torch.cat((self.y, next_chars), dim=-1)

        # VERY IMPORTANT! we need a mask for
        # the log likelihood when reaching the eos
        # self.ll_mask = torch.zeros(len(self.loglikelihood), dtype=torch.bool)
        self.ll_mask = torch.any(self.y == self.tokenizer["end"], dim=-1)

    def _get_topk(self, loglikelihood: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = loglikelihood.shape[-1]
        if self.n_predictions == 1:  # Greedy sampling
            loglikelihood, next_chars = loglikelihood.max(axis=-1)
        else:
            loglikelihood, next_chars = loglikelihood.topk(k=min(v, self.n_predictions), axis=-1)

        if v < self.n_predictions:
            d = self.n_predictions - len(self.tokenizer)
            pl = -1e20 * torch.ones(
                (len(loglikelihood), d),
                dtype=loglikelihood.dtype,
                device=loglikelihood.device,
            )
            pc = torch.zeros(
                (len(next_chars), d),
                dtype=next_chars.dtype,
                device=loglikelihood.device,
            )
            loglikelihood = torch.cat((loglikelihood, pl), dim=-1)
            next_chars = torch.cat((next_chars, pc), dim=-1)
        return loglikelihood, next_chars


class FastBeamSearchSampler(BeamSearchSampler):

    sampler_name = "FastBeamSearchSampler"

    """
    Faster sampling, but less memory efficient than BeamSearchSampler.
    """

    def __init__(self, tokenizer: SMILESTokenizer, device: Union[torch.device, str], batch_size: int):
        super().__init__(tokenizer, device, batch_size)

    def get_actions(self) -> torch.Tensor:
        next_loglikelihood = []

        X = {"decoder_input": self.y, "memory_input": self.x, "memory_pad_mask": self.x_mask}

        with torch.no_grad():
            ll = self.model.decode_batch(X)

        next_loglikelihood.append(ll)
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()
        return next_loglikelihood


class MultinomialSampler(BaseSamplerNode):

    sampler_name = "MultinomialSampler"

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        device: Union[torch.device, str],
        batch_size: int,
        temperature: float = 0.1,
    ):
        self._temperature = temperature
        super().__init__(tokenizer, device, batch_size)

    def _init_action(self, loglikelihood: torch.Tensor) -> None:
        # Perform the first step
        loglikelihood, next_chars = self._multinomial_sample(loglikelihood, self.n_predictions)

        self.loglikelihood = loglikelihood.view(-1, 1)
        next_chars = next_chars.view(-1, 1)

        self.y = self.y.view(len(self.y), 1, -1).repeat(1, self.n_predictions, 1).view(-1, 1)
        self.x = self.x[:, None].repeat(1, self.n_predictions, 1, 1).view((-1,) + tuple(self.x.shape[1:]))
        self.x_mask = self.x_mask[:, None].repeat(1, self.n_predictions, 1).view((-1,) + tuple(self.x_mask.shape[1:]))

        self.y = torch.cat((self.y, next_chars), dim=-1)

        # VERY IMPORTANT! we need a mask for
        # the log likelihood when reaching the eos
        # self.ll_mask = torch.zeros(len(self.loglikelihood), dtype=torch.bool)
        self.ll_mask = torch.any(self.y == self.tokenizer["end"], dim=-1)

    def action(self, next_loglikelhihood: torch.Tensor) -> None:
        if self.pos == 0:
            self._init_action(next_loglikelhihood)
        else:
            vocabulary_size = len(self.tokenizer)
            # set loglikehihood to the maxium (0)
            # when observed an eos_token
            next_loglikelhihood[self.ll_mask, :] = (
                torch.minimum(self.loglikelihood.min(), next_loglikelhihood.min()) - 1.0
            )
            next_loglikelhihood[self.ll_mask, self.tokenizer["end"]] = 0.0
            # done

            ll = self.loglikelihood + next_loglikelhihood
            ll, idx = self._multinomial_sample(ll)

            # tricky indexing
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            if best_candidates.device != self.device:
                best_candidates = best_candidates.to(self.device)
            # done

            self.y = torch.cat((self.y, next_chars), dim=-1)
            self.loglikelihood = ll.view(-1, 1)

            # update ll mask
            self.ll_mask = torch.any(self.y == self.tokenizer["end"], dim=-1)
        self.pos = self.pos + 1

    def _multinomial_sample(self, loglikelihood: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        likelihood = torch.nn.functional.softmax(loglikelihood / self._temperature, dim=1)
        next_chars = likelihood.multinomial(n_samples, replacement=False)
        loglikelihood = torch.gather(loglikelihood, dim=-1, index=next_chars)
        return loglikelihood, next_chars
