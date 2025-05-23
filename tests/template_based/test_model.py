import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from aizynthmodels.template_based.models.models import MulticlassClassifier
from aizynthmodels.utils.scores import ScoreCollection

random.seed(a=1)
torch.manual_seed(1)


def test_forward_shape(model_hyperparams, dummy_data):

    batch_size = 3
    model = MulticlassClassifier(model_hyperparams)

    inputs, labels = dummy_data

    n_classes = labels.shape[1]
    batch_input = torch.from_numpy(inputs[0:batch_size].toarray().astype(np.float32))
    output = model(batch_input)

    assert tuple(output.shape) == (batch_size, n_classes)


def test_sample_predictions(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    batch_size = 3

    model = MulticlassClassifier(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size, :].toarray().astype(np.float32)),
    }

    predictions = model.sample_predictions(batch_input, n_predictions=2)

    probabilities_expected = [[0.59, 0.41], [0.59, 0.41], [0.59, 0.41]]

    assert len(predictions["predictions"]) == 3
    assert len(predictions["predictions"][0]) == 2
    assert all(
        [
            round(float(prob), 2) == prob_exp
            for prob_top_n, prob_exp_top_n in zip(predictions["probabilities"], probabilities_expected)
            for prob, prob_exp in zip(prob_top_n, prob_exp_top_n)
        ]
    )


def test_sample_predictions_templates(tmpdir, model_hyperparams, dummy_data, reaction_data):
    pl.seed_everything(1)

    batch_size = 3

    templates = pd.DataFrame({"template_code": np.arange(len(reaction_data[2])), "retro_template": reaction_data[2]})

    templates_file = tmpdir / "dummy_unique_templates.csv"
    templates.to_csv(templates_file, sep="\t", index=False)

    model = MulticlassClassifier(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size, :].toarray().astype(np.float32)),
        "reactant": reaction_data[0][0:batch_size],
        "product": reaction_data[1][0:batch_size],
    }

    model.set_templates(templates_file)
    predictions = model.sample_predictions(batch_input, n_predictions=2)

    probabilities_expected = [[0.59, 0.41], [0.59, 0.41], [0.59, 0.41]]
    assert predictions["predictions"] == [
        [["CCS(=O)(=O)Cl.OCCBr"], ["dummy-smiles"]],
        [["CS(=O)(=O)Cl.OCCCBr"], ["dummy-smiles"]],
        [["CC(C)CS(=O)(=O)Cl.OCCCl"], ["dummy-smiles"]],
    ]

    assert all(
        [
            round(float(prob), 2) == prob_exp
            for prob_top_n, prob_exp_top_n in zip(predictions["probabilities"], probabilities_expected)
            for prob, prob_exp in zip(prob_top_n, prob_exp_top_n)
        ]
    )


def test_train_step(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    batch_size = 3

    model = MulticlassClassifier(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size, :].toarray().astype(np.float32)),
    }

    model.eval()

    loss = model.training_step(batch_input)
    assert round(loss.cpu().item(), 4) == 0.8992


def test_val_step(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    scores = ScoreCollection()
    scores.load_from_config([{"TopKAccuracyScore": [{"canonicalized": True, "top_ks": [1, 3, 5, 10]}]}])

    batch_size = 3
    n_classes = 50
    model_hyperparams["num_classes"] = n_classes

    model = MulticlassClassifier(model_hyperparams)
    inputs, labels = dummy_data

    labels = labels.toarray().astype(np.float32)
    labels_aug = np.zeros((batch_size, n_classes), np.float32)
    labels_aug[:, 0:2] = labels[2 : batch_size + 2, :]  # noqa: E203

    batch_input = {
        "input": torch.from_numpy(inputs[2 : batch_size + 2, :].toarray().astype(np.float32)),  # noqa: E203
        "label": torch.from_numpy(labels_aug),
    }

    model.scores = scores
    model.eval()

    metrics = model.validation_step(batch_input, None)
    assert list(metrics.keys()) == ["accuracy_top_1", "accuracy_top_3", "accuracy_top_5", "accuracy_top_10"]
    assert round(metrics["accuracy_top_10"], 2) == 0.33


def test_test_step(model_hyperparams, dummy_data):
    pl.seed_everything(1)

    scores = ScoreCollection()
    scores.load_from_config([{"TopKAccuracyScore": [{"canonicalized": True}]}])

    batch_size = 3

    model = MulticlassClassifier(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "input": torch.from_numpy(inputs[2 : batch_size + 2, :].toarray().astype(np.float32)),  # noqa: E203
        "label": torch.from_numpy(labels[2 : batch_size + 2, :].toarray().astype(np.float32)),  # noqa: E203
    }

    model.n_predictions = 2
    model.scores = scores
    model.eval()

    metrics = model.test_step(batch_input, None)
    assert round(metrics["accuracy_top_1"], 2) == 0.67


def test_test_step_with_templates(tmpdir, model_hyperparams, dummy_data, reaction_data):
    pl.seed_everything(1)
    batch_size = 3

    templates = pd.DataFrame({"template_code": np.arange(len(reaction_data[2])), "retro_template": reaction_data[2]})

    templates_file = tmpdir / "dummy_unique_templates.csv"
    templates.to_csv(templates_file, sep="\t", index=False)

    model = MulticlassClassifier(model_hyperparams)
    inputs, labels = dummy_data

    batch_input = {
        "input": torch.from_numpy(inputs[0:batch_size, :].toarray().astype(np.float32)),
        "label": torch.from_numpy(labels[0:batch_size, :].toarray().astype(np.float32)),
        "reactant": reaction_data[0][0:batch_size],
        "product": reaction_data[1][0:batch_size],
    }

    model.set_templates(templates_file)

    scores = ScoreCollection()
    scores.load_from_config([{"TopKAccuracyScore": [{"canonicalized": True}]}, "FractionInvalidScore"])

    model.n_predictions = 2
    model.scores = scores
    model.eval()

    metrics = model.test_step(batch_input, None)

    assert round(metrics["accuracy_top_1"], 2) == 1.0
    assert round(metrics["fraction_invalid"], 2) == 0.5
    assert metrics["ground_truth"] == ["CCS(=O)(=O)Cl.OCCBr", "CS(=O)(=O)Cl.OCCCBr", "CC(C)CS(=O)(=O)Cl.OCCCl"]
