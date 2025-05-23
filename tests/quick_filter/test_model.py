import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from aizynthmodels.quick_filter.models.models import ClassificationModel
from aizynthmodels.utils.scores import ScoreCollection

random.seed(a=1)
torch.manual_seed(1)


def test_forward_shape(model_hyperparams, dummy_data):

    batch_size = 3
    model = ClassificationModel(model_hyperparams)

    inputs, _ = dummy_data

    batch_input = torch.from_numpy(inputs[0:batch_size].toarray().astype(np.float32))
    output = model(batch_input, batch_input)

    assert tuple(output.shape) == (batch_size,)


def test_sample_predictions(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    batch_size = 3
    model_hyperparams["threshold"] = 0.0

    model = ClassificationModel(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "product_input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "reaction_input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size].astype(np.float32)),
    }

    predictions = model.sample_predictions(batch_input, n_predictions=2)

    assert len(predictions["predictions"]) == 3
    assert predictions["predictions"] == [1.0, 1.0, 1.0]


def test_train_step(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    batch_size = 3
    model_hyperparams["threshold"] = 0.0

    model = ClassificationModel(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "product_input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "reaction_input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size].astype(np.float32)),
    }

    model.eval()

    loss = model.training_step(batch_input)
    assert round(loss.cpu().item(), 4) == 0.6856


def test_val_step(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    scores = ScoreCollection()
    scores.load_from_config([{"BinaryAccuracyScore": []}])

    batch_size = 3
    model_hyperparams["threshold"] = 0.0

    model = ClassificationModel(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "product_input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "reaction_input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size].astype(np.float32)),
    }

    model.scores = scores
    model.eval()

    metrics = model.validation_step(batch_input, None)
    assert list(metrics.keys()) == ["binary_accuracy"]
    assert round(metrics["binary_accuracy"], 2) == 0.33


def test_test_step(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    scores = ScoreCollection()
    scores.load_from_config([{"BinaryAccuracyScore": []}])

    batch_size = 3
    model_hyperparams["threshold"] = 0.0

    model = ClassificationModel(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "product_input": torch.from_numpy(inputs[2 : batch_size + 2, :].toarray().astype(np.float32)),  # noqa: E203
        "reaction_input": torch.from_numpy(inputs[2 : batch_size + 2, :].toarray().astype(np.float32)),  # noqa: E203
        "label": torch.from_numpy(labels[2 : batch_size + 2].astype(np.float32)),  # noqa: E203
    }

    model.n_predictions = 2
    model.scores = scores
    model.eval()

    metrics = model.test_step(batch_input, None)
    assert round(metrics["binary_accuracy"], 2) == 0.67
