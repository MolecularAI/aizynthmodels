import numpy as np
import pytest

from aizynthmodels.chemformer.sampler import SMILESSampler


# Test BeamSearchSampler
@pytest.mark.parametrize(
    ("beam_size"),
    [(1), (3), (5)],
)
def test_beam_search_sampler(model_batch_setup, beam_size):
    chemformer = model_batch_setup["chemformer"]

    batch_input = chemformer.on_device(model_batch_setup["batch_input"])

    sampled_smiles, llhs = chemformer.sampler.sample_molecules(chemformer.model, batch_input, beam_size)

    assert len(llhs[0]) == beam_size
    assert len(sampled_smiles[0]) == beam_size
    assert len(sampled_smiles) == batch_input["encoder_input"].shape[1]
    assert len(sampled_smiles) == len(llhs)


# Test FastBeamSearchSampler
@pytest.mark.parametrize(
    ("beam_size"),
    [(1), (3), (5)],
)
def test_fast_beam_search_sampler(model_batch_setup, beam_size):
    chemformer = model_batch_setup["chemformer"]
    chemformer.sampler = SMILESSampler(
        chemformer.tokenizer,
        chemformer.config.model_hyperparams.max_seq_len,
        device="cpu",
        sampler_node="FastBeamSearchSampler",
    )
    batch_input = chemformer.on_device(model_batch_setup["batch_input"])

    sampled_smiles, llhs = chemformer.sampler.sample_molecules(chemformer.model, batch_input, beam_size)

    assert len(llhs[0]) == beam_size
    assert len(sampled_smiles[0]) == beam_size
    assert len(sampled_smiles) == batch_input["encoder_input"].shape[1]
    assert len(sampled_smiles) == len(llhs)


# Test MultinomialSampler
@pytest.mark.parametrize(
    ("n_predictions"),
    [(1), (3), (5)],
)
def test_multinomial_sampler(model_batch_setup, n_predictions):

    chemformer = model_batch_setup["chemformer"]
    chemformer.sampler = SMILESSampler(
        chemformer.tokenizer,
        chemformer.config.model_hyperparams.max_seq_len,
        device="cpu",
        sampler_node="MultinomialSampler",
    )
    batch_input = chemformer.on_device(model_batch_setup["batch_input"])

    sampled_smiles, llhs = chemformer.sampler.sample_molecules(chemformer.model, batch_input, n_predictions)

    assert len(llhs[0]) == n_predictions
    assert len(sampled_smiles[0]) == n_predictions
    assert len(sampled_smiles) == batch_input["encoder_input"].shape[1]
    assert len(sampled_smiles) == len(llhs)
    # Make sure the loglikelihoods are sorted
    llhs_sorted = np.sort(llhs[0])[::-1]
    assert all(llhs[0] == llhs_sorted)


def test_return_tokenized(model_batch_setup):
    chemformer = model_batch_setup["chemformer"]

    batch_input = chemformer.on_device(model_batch_setup["batch_input"])

    beam_size = 2
    sampled_tokens, llhs = chemformer.sampler.sample_molecules(
        chemformer.model, batch_input, beam_size, return_tokenized=True
    )

    assert len(llhs) == batch_input["encoder_input"].shape[1]
    assert len(llhs[0]) == beam_size
    assert len(sampled_tokens[0]) == chemformer.config.model_hyperparams.max_seq_len
