import random

import torch

from aizynthmodels.chemformer.models.bart import BARTModel
from aizynthmodels.chemformer.utils.models import (
    calculate_token_accuracy,
    construct_embeddings,
    generate_square_subsequent_mask,
    positional_embeddings,
    set_model_hyperparams,
)

random.seed(a=1)
torch.manual_seed(1)


def test_set_model_hyperparams(default_config, setup_tokenizer):
    hyperparams = set_model_hyperparams(default_config, setup_tokenizer(), 0)
    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["pad_token_idx"] = 0
    model_hyperparams["num_steps"] = 0
    model_hyperparams["vocabulary_size"] = 15
    model_hyperparams["task"] = "forward_prediction"

    assert hyperparams == model_hyperparams


def test_set_model_hyperparams_train_mode(default_config, setup_tokenizer):
    default_config.mode = "train"
    default_config.model_path = "test"
    default_config.resume = True
    hyperparams = set_model_hyperparams(default_config, setup_tokenizer(), 0)

    assert hyperparams["num_steps"] == 1


def test_calculate_token_accuracy(setup_encoder, reactant_data, default_config):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    react_ids, react_mask = encoder(reactant_data[1:])
    target_ids = react_ids[1:, :]
    target_mask = react_mask[1:, :]

    token_output = torch.rand([8, len(reactant_data[1:]), len(tokenizer)])

    # Batch element 0
    token_output[0, 0, 6] += 1
    token_output[1, 0, 6] -= 1
    token_output[2, 0, 9] += 1
    token_output[3, 0, 3] += 1
    token_output[4, 0, 0] += 1
    token_output[5, 0, 0] -= 1

    # Batch element 1
    token_output[0, 1, 6] += 1
    token_output[1, 1, 10] += 1
    token_output[2, 1, 11] += 1
    token_output[3, 1, 7] += 1
    token_output[4, 1, 12] -= 1
    token_output[5, 1, 6] += 1
    token_output[6, 1, 13] -= 1
    token_output[7, 1, 3] += 1

    batch_input = {"target": target_ids, "target_mask": target_mask}
    model_output = {"token_output": token_output}
    token_acc = calculate_token_accuracy(batch_input, model_output)

    exp_token_acc = (3 + 6) / (4 + 8)

    assert exp_token_acc == token_acc


def test_construct_input_shape(setup_encoder, default_config, reactant_data):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model = BARTModel(**model_hyperparams)

    token_ids, _ = encoder(reactant_data)

    emb = construct_embeddings(
        token_ids,
        model.hparams.d_model,
        model.hparams.batch_first,
        model.emb,
        positional_embeddings(model.hparams.d_model, model.hparams.max_seq_len),
    )

    assert emb.shape[0] == max([len(ts) for ts in token_ids.transpose(0, 1)])
    assert emb.shape[1] == 3
    assert emb.shape[2] == model_hyperparams["d_model"]


def test_generate_square_subsequent_mask():
    mask = generate_square_subsequent_mask(1)

    assert mask == 0


def test_positional_embeddings(default_config):
    pos_embs = positional_embeddings(
        default_config.model_hyperparams["d_model"],
        default_config.model_hyperparams["max_seq_len"],
    )

    assert pos_embs.shape[0] == default_config.model_hyperparams["max_seq_len"]
    assert pos_embs.shape[1] == default_config.model_hyperparams["d_model"]
    assert pos_embs.tolist()[0] == [0.0, 1.0, 0.0, 1.0]
