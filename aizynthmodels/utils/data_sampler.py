"""Utility classes for sampling PyTorch datasets."""
from typing import Iterator, List, Union

import numpy as np
from torch.utils.data import BatchSampler, Sampler


class ChunkBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Union[Sampler[int]],
        batch_size: int,
        drop_last: bool,
        i_chunk: int = 0,
        n_chunks: int = 1,
        **kwargs
    ) -> None:
        """
        A sampler which only samples a specific chunk of batches.

        :param sampler: The torch.Sampler used to sample data indices.
        :param batch_size: the number of samples in a batch.
        :param drop_last: whether to keep or drop the last batch (the last batch might
            be smaller than the other batches)
        :param i_chunk: the index of the current chunk of batches.
        :param n_chunks: the total number of chunks to divide the batches into.
        """
        super().__init__(sampler, batch_size, drop_last, **kwargs)
        self.i_chunk = i_chunk
        self.n_chunks = n_chunks
        self._batch_counter = 0
        self._set_start_end_batches()

    def __iter__(self) -> Iterator[List[int]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while self._batch_counter < self.end_batch_idx:
                batch = [next(sampler_iter) for _ in range(self.batch_size)]

                if self._batch_counter < self.start_batch_idx:
                    self._batch_counter += 1
                    continue

                self._batch_counter += 1
                yield batch
        else:
            batch = [0] * int(self.batch_size)
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    if self._batch_counter < self.start_batch_idx:
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                        self._batch_counter += 1
                        continue

                    self._batch_counter += 1
                    if self._batch_counter >= self.end_batch_idx:
                        break

                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
        self._batch_counter = 0

    def __len__(self):
        return self.end_batch_idx - self.start_batch_idx

    def _set_start_end_batches(self) -> None:
        """Divide batches into chunks of batches"""
        n_batches = int(np.ceil(len(self.sampler) / self.batch_size))
        n_batches_in_chunk = int(n_batches / float(self.n_chunks))
        self.start_batch_idx = self.i_chunk * n_batches_in_chunk
        if self.i_chunk != self.n_chunks - 1:
            self.end_batch_idx = self.start_batch_idx + n_batches_in_chunk
        else:
            if self.drop_last:
                self.end_batch_idx = n_batches - 1
            else:
                self.end_batch_idx = n_batches
