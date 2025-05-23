import numpy as np
import pytest
import torch

from aizynthmodels.utils.data_sampler import ChunkBatchSampler


@pytest.fixture
def data():
    return torch.arange(1000)


def test_chunk_batch_sampler(data):
    sampler = torch.utils.data.SequentialSampler(data)
    batch_size = 64
    drop_last = False
    i_chunk = 0
    n_chunks = 1

    chunk_sampler = ChunkBatchSampler(sampler, batch_size, drop_last, i_chunk, n_chunks)

    assert len(chunk_sampler) == np.ceil(data.shape[0] / batch_size)

    i_chunk = 0
    n_chunks = 3

    chunk_sampler = ChunkBatchSampler(sampler, batch_size, drop_last, i_chunk, n_chunks)

    assert len(chunk_sampler) == np.floor(data.shape[0] / (batch_size * n_chunks))

    i_chunk = 2
    n_chunks = 3

    chunk_sampler = ChunkBatchSampler(sampler, batch_size, drop_last, i_chunk, n_chunks)

    assert len(chunk_sampler) == np.ceil(data.shape[0] / (batch_size * n_chunks))


def test_chunk_batch_iterator(data):
    sampler = torch.utils.data.SequentialSampler(data)
    batch_size = 64
    drop_last = False
    i_chunk = 0
    n_chunks = 3

    chunk_sampler = ChunkBatchSampler(sampler, batch_size, drop_last, i_chunk, n_chunks)

    samples = []
    for x in chunk_sampler:
        samples.append(x)

    assert len(samples) == 5
    assert all(idx == idx_expected for idx, idx_expected in zip(samples[-1], range(256, 320)))


def test_chunk_batch_sampler_drop_last(data):
    sampler = torch.utils.data.SequentialSampler(data)
    batch_size = 64
    i_chunk = 2
    n_chunks = 3

    drop_last = False
    sampler_no_drop = ChunkBatchSampler(sampler, batch_size, drop_last, i_chunk, n_chunks)

    drop_last = True
    sampler_drop = ChunkBatchSampler(sampler, batch_size, drop_last, i_chunk, n_chunks)

    assert len(sampler_drop) == len(sampler_no_drop) - 1

    samples1 = []
    samples2 = []
    for x in sampler_no_drop:
        samples1.append(x)

    for y in sampler_drop:
        samples2.append(y)

    assert len(samples2) == len(samples1) - 1
