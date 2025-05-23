import random

import numpy as np
import torch

from aizynthmodels.chemformer.models.bart import BARTModel

random.seed(a=1)
torch.manual_seed(1)


def test_forward_shape(default_config, setup_encoder, reactant_data, product_data):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    model = BARTModel(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data, mask=True)
    prod_ids, prod_mask = encoder(product_data, mask=True)

    batch_input = {
        "encoder_input": react_ids,
        "encoder_pad_mask": react_mask,
        "decoder_input": prod_ids,
        "decoder_pad_mask": prod_mask,
    }

    output = model(batch_input)
    model_output = output["model_output"]
    token_output = output["token_output"]

    exp_seq_len = 4  # From expected tokenised length of prod data
    exp_batch_size = len(product_data)
    exp_dim = model_hyperparams["d_model"]
    exp_vocab_size = len(tokenizer)

    assert tuple(model_output.shape) == (exp_seq_len, exp_batch_size, exp_dim)
    assert tuple(token_output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)


def test_encode_shape(default_config, setup_encoder, reactant_data):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model = BARTModel(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data)

    batch_input = {"encoder_input": react_ids, "encoder_pad_mask": react_mask}

    output = model.encode(batch_input)

    exp_seq_len = 9  # From expected tokenised length of react data
    exp_batch_size = len(reactant_data)
    exp_dim = model_hyperparams["d_model"]

    if model.hparams.batch_first:
        assert tuple(output.shape) == (exp_batch_size, exp_seq_len, exp_dim)
    else:
        assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_dim)


def test_decode_shape(default_config, setup_encoder, reactant_data, product_data):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    model = BARTModel(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data)
    encode_input = {"encoder_input": react_ids, "encoder_pad_mask": react_mask}

    memory = model.encode(encode_input)

    prod_ids, prod_mask = encoder(product_data)
    batch_input = {
        "decoder_input": prod_ids,
        "decoder_pad_mask": prod_mask,
        "memory_input": memory,
        "memory_pad_mask": react_mask,
    }

    decoder_output = model.decode(batch_input)
    output = model.generator(decoder_output)

    exp_seq_len = 4  # From expected tokenised length of prod data
    exp_batch_size = len(reactant_data)
    exp_vocab_size = len(tokenizer)

    assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)


def test_batch_first(model_batch_setup, model_batch_setup_batch_first):
    beam_size = 3
    chemformer = model_batch_setup["chemformer"]
    chemformer_bf = model_batch_setup_batch_first["chemformer"]

    chemformer_bf.model.load_state_dict(chemformer.model.state_dict())  # Copy the weights

    assert not chemformer.config.model_hyperparams["batch_first"]
    assert chemformer_bf.config.model_hyperparams["batch_first"]

    batch_input = chemformer.on_device(model_batch_setup["batch_input"])

    sampled_smiles, llhs = chemformer.sampler.sample_molecules(chemformer.model, batch_input, beam_size)
    sampled_smiles_bf, llhs_bf = chemformer_bf.sampler.sample_molecules(chemformer_bf.model, batch_input, beam_size)

    assert all(all(is_equal) for is_equal in np.equal(llhs, llhs_bf))
    assert all(all(is_equal) for is_equal in np.equal(sampled_smiles, sampled_smiles_bf))


def test_sample_molecules(model_batch_setup):
    """Test that the batch first model produces the same output as the non-batch-first model."""
    chemformer = model_batch_setup["chemformer"]

    batch_input = chemformer.on_device(model_batch_setup["batch_input"])

    sampled_smiles, llhs = chemformer.model.sample_predictions(batch_input)

    llhs_expected = [
        [-1155.52, -1155.72, -1155.77],
        [-1146.84, -1147.02, -1147.08],
        [-1161.98, -1161.98, -1162.02],
    ]

    assert len(sampled_smiles) == 3
    assert all(
        [
            round(llh, 2) == llh_exp
            for llh_top_n, llh_exp_top_n in zip(llhs, llhs_expected)
            for llh, llh_exp in zip(llh_top_n, llh_exp_top_n)
        ]
    )


def test_loss(model_batch_setup):
    batch_input = model_batch_setup["batch_input"]
    model = model_batch_setup["chemformer"].model

    model_output = model(batch_input)
    loss = model.loss(batch_input, model_output)

    assert round(loss.tolist(), 4) == 3.2243
