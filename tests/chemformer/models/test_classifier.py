import random

import pytorch_lightning as pl
import torch

from aizynthmodels.chemformer.models.classifier import TransformerClassifier
from aizynthmodels.utils.scores import ScoreCollection

random.seed(a=1)
torch.manual_seed(1)


def test_forward_shape(default_config, setup_encoder, reactant_data):
    tokenizer, _, encoder = setup_encoder

    batch_size = 2
    default_config.batch_size = batch_size
    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model_hyperparams["num_hidden_nodes"] = [64, 32]

    model = TransformerClassifier(**model_hyperparams)

    smiles_ids, smiles_mask = encoder(reactant_data[0:batch_size], mask=True)

    batch_input = {
        "encoder_input": smiles_ids,
        "encoder_pad_mask": smiles_mask,
    }

    output = model(batch_input)
    model_output = output["logits"]

    assert tuple(model_output.shape) == (batch_size, default_config.model_hyperparams.num_classes)


def test_encode_shape(default_config, setup_encoder, reactant_data):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model_hyperparams["num_hidden_nodes"] = []
    model = TransformerClassifier(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data)

    batch_input = {"encoder_input": react_ids, "encoder_pad_mask": react_mask}

    output = model.encode(batch_input)

    exp_batch_size = len(reactant_data)
    exp_dim = model_hyperparams["d_model"]

    assert tuple(output.shape) == (exp_batch_size, exp_dim)


def test_sample_predictions(default_config, setup_encoder, reactant_data):
    pl.seed_everything(1)
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    model = TransformerClassifier(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data)
    batch_input = {"encoder_input": react_ids, "encoder_pad_mask": react_mask}

    model.n_predictions = 3
    predictions, probabilities = model.sample_predictions(batch_input)

    probabilities_expected = [
        [0.47, 0.30, 0.23],
        [0.44, 0.35, 0.21],
        [0.46, 0.30, 0.24],
    ]

    assert len(predictions) == 3
    assert len(predictions[0]) == model.n_predictions
    assert all(
        [
            round(float(prob), 2) == prob_exp
            for prob_top_n, prob_exp_top_n in zip(probabilities, probabilities_expected)
            for prob, prob_exp in zip(prob_top_n, prob_exp_top_n)
        ]
    )


def test_loss(default_config, setup_encoder, reactant_data):
    pl.seed_everything(1)
    tokenizer, _, encoder = setup_encoder
    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    model = TransformerClassifier(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data)
    batch_input = {
        "encoder_input": react_ids,
        "encoder_pad_mask": react_mask,
        "class_indicator": torch.Tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    }

    model_output = model(batch_input)
    loss = model.loss(batch_input, model_output)
    assert round(loss.tolist(), 4) == 1.1742


def test_test_step(default_config, setup_encoder, reactant_data):
    pl.seed_everything(1)
    tokenizer, _, encoder = setup_encoder
    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    model = TransformerClassifier(**model_hyperparams)

    react_ids, react_mask = encoder(reactant_data)
    batch_input = {
        "encoder_input": react_ids,
        "encoder_pad_mask": react_mask,
        "label": [0, 2, 1],
        "class_indicator": torch.Tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    }

    scores = ScoreCollection()
    scores.load_from_config([{"TopKAccuracyScore": [{"canonicalized": True}]}])
    model.scores = scores
    model.n_predictions = 3
    model.test_step(batch_input, 0)

    assert round(model.test_step_outputs[0]["test_loss"], 4) == 1.1742
