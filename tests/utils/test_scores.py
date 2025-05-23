import numpy as np
import omegaconf as oc
import pytest
import torch

from aizynthmodels.utils.configs.chemformer.inference_score import InferenceScore
from aizynthmodels.utils.scores import (
    AveragePrecisionScore,
    BalancedAccuracyScore,
    BaseScore,
    FractionInvalidScore,
    FractionUniqueScore,
    MeanAbsoluteError,
    R2Score,
    RecallScore,
    ScoreCollection,
    TanimotoSimilarityScore,
    TopKAccuracyScore,
    TopKCoverageScore,
)
from aizynthmodels.utils.trainer import instantiate_scorers


@pytest.fixture
def smiles_predictions():
    sampled_smiles = [
        ["CCO", "CCCO", "CCO", "CCCO", "CCO"],
        ["c1ccc1", "c1cccc1", "c1cccc1", "c1ccccc1", "c1ccccc1"],
        ["CCO", "CO", "CO", "CO", "CO"],
    ]
    target_smiles = ["CCO", "c1ccccc1", "CO"]
    return {"sampled_smiles": sampled_smiles, "target_smiles": target_smiles}


@pytest.fixture
def smiles_predictions_grouped():
    sampled_smiles = [
        [["CCO"], ["CCCO"], ["CCO"], ["CCCO"], ["CCO"]],
        [["c1ccc1", "CO"], ["c1cccc1"], ["c1cccc1"], ["CO", "c1ccccc1"], ["c1ccccc1"]],
        [["CCCO", "CCO"], ["CCO", "CO"], ["CO"], ["CO"], ["CO"]],
    ]
    target_smiles = ["CCO", "c1ccccc1", "CO"]
    return {"sampled_smiles": sampled_smiles, "target_smiles": target_smiles}


@pytest.fixture
def torch_predictions_class():
    predictions = torch.from_numpy(np.array([[1, 2, 1], [2, 3, 3], [3, 1, 1], [3, 1, 4]]))
    ground_truth = torch.from_numpy(np.array([1, 3, 2, 4]))
    return {"predictions": predictions, "ground_truth": ground_truth}


@pytest.fixture
def torch_predictions_float():
    predictions = torch.from_numpy(np.array([0.1, 0.25, 0.15]))
    ground_truth = torch.from_numpy(np.array([0.12, 0.22, 0.14]))
    return {"predictions": predictions, "ground_truth": ground_truth}


@pytest.fixture
def torch_predictions_binary_class():
    predictions = torch.tensor([0, 1, 0, 1])
    ground_truth = torch.tensor([0, 1, 1, 0])
    return {"predictions": predictions, "ground_truth": ground_truth}


def test_instantiate_scorers():
    config = oc.OmegaConf.structured(InferenceScore)
    scorers = instantiate_scorers(config.get("scorers"))

    assert isinstance(scorers, ScoreCollection)
    assert len(scorers.names()) == 4

    scorer_names = set(scorers.names())
    expected = set(
        [
            "top_k_accuracy",
            "fraction_invalid",
            "top1_tanimoto_similarity",
            "fraction_unique",
        ]
    )

    assert expected.issubset(scorer_names)

    scorers = instantiate_scorers(None)
    assert isinstance(scorers, ScoreCollection)
    assert len(scorers.names()) == 0


def test_default_inference_scoring():
    config = oc.OmegaConf.structured(InferenceScore)
    score_config = config.get("scorers")
    scorers = instantiate_scorers(score_config)

    sampled_smiles = [["C!O", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]]
    target_smiles = ["CCO", "c1ccccc1"]

    metrics_scores = scorers.apply(sampled_smiles, target_smiles)
    assert round(metrics_scores["fraction_invalid"], 4) == 0.3333
    assert round(metrics_scores["fraction_unique"], 4) == 0.3333
    assert metrics_scores["top1_tanimoto_similarity"] == 1.0
    assert metrics_scores["accuracy_top_1"] == 0.5
    assert metrics_scores["accuracy_top_3"] == 1.0


def test_num_samples_not_num_targets():
    scorer = ScoreCollection()
    scorer.load(FractionInvalidScore())

    with pytest.raises(ValueError, match="The number of predictions and targets must be the same, got 2 and 3"):
        scorer.apply([["C", "CC"], ["CO", "CO"]], ["CC", "CO", "CCO"])


def test_base_score():
    scorer = ScoreCollection()
    with pytest.raises(
        ValueError, match="Only objects of classes inherited from aizynthmodels.utils.scores.BaseScore can be added"
    ):
        scorer.load(BaseScore)

    score = BaseScore()
    predictions = np.array([3, 1, 2])
    ground_truth = np.array([3, 2, 2])
    with pytest.raises(NotImplementedError):
        score(torch.from_numpy(predictions), torch.from_numpy(ground_truth))


@pytest.mark.parametrize(
    ("sampled_smiles", "target_smiles", "expected_score"),
    [
        (
            [["CCO", "CCO", "CCO"], ["c1cc!ccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.3333,
        ),
        (
            [["CCO", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.1667,
        ),
        (
            [["CCO", "C!O", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.3333,
        ),
        (
            [["CCO", "CCO", "CCO"], ["c1ccccc1", "c1ccccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.0,
        ),
        (
            [[["CCO"], ["CC!O"], ["CCO"]], [["c1ccccc1"], ["c1ccccc1"], ["c1ccccc1"]]],  # Grouped predictions
            ["CCO", "c1ccccc1"],
            0.1667,
        ),
        (
            [
                [["CCO"], ["CCO", "CC!O"], ["CCO"]],
                [["c1ccccc1"], ["c1ccccc1"], ["c1ccccc1"]],
            ],  # One valid prediction per group
            ["CCO", "c1ccccc1"],
            0.0,
        ),
    ],
)
def test_fraction_invalid(sampled_smiles, target_smiles, expected_score):
    scorer = ScoreCollection()
    scorer.load(FractionInvalidScore())

    assert isinstance(scorer.objects()[0], FractionInvalidScore)

    score = scorer.apply(sampled_smiles, target_smiles)["fraction_invalid"]
    assert round(score, 4) == expected_score


@pytest.mark.parametrize(
    ("sampled_smiles", "target_smiles", "expected_score"),
    [
        (
            [["CCO", "CCO", "CCO"], ["c1cc!ccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.5,
        ),
        (
            [["CCO", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.0,
        ),
    ],
)
def test_fraction_invalid_only_top1(sampled_smiles, target_smiles, expected_score):
    scorer = ScoreCollection()
    scorer.load(FractionInvalidScore(only_top1=True))

    score = scorer.apply(sampled_smiles, target_smiles)["fraction_invalid"]
    assert round(score, 4) == expected_score


@pytest.mark.parametrize(
    ("sampled_smiles", "target_smiles", "expected_score"),
    [
        (
            [["CCO", "CCO", "CCO"], ["c1cc!ccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.3333,
        ),
        (
            [["CCO", "CCO", "CCO"], ["c1ccccc1", "c1cccc1", "c1ccccc1"]],
            ["CCO", "c1ccccc1"],
            0.3333,
        ),
        (
            [["CCO", "C!O", "COO"], ["c1ccccc1", "c1cccc1", "c1cc(Br)ccc1"]],
            ["CCO", "c1ccccc1"],
            0.6667,
        ),
    ],
)
def test_fraction_unique(sampled_smiles, target_smiles, expected_score):
    scorer = ScoreCollection()
    scorer.load(FractionUniqueScore())

    score = scorer.apply(sampled_smiles, target_smiles)["fraction_unique"]
    assert round(score, 4) == expected_score


@pytest.mark.parametrize(
    ("sampled_smiles", "target_smiles", "expected_score"),
    [
        (
            [["CCO", "C!O", "COO"], ["c1ccccc1", "c1cccc1", "c1cc(Br)ccc1"]],
            ["CCO", "c1ccccc1"],
            0.6667,
        ),
    ],
)
def test_fraction_unique_canonical(sampled_smiles, target_smiles, expected_score):
    scorer = ScoreCollection()
    scorer.load(FractionUniqueScore(canonicalized=True))

    score = scorer.apply(sampled_smiles, target_smiles)["fraction_unique"]
    assert round(score, 4) == expected_score


def test_coverage(smiles_predictions):
    scorer = ScoreCollection()
    scorer.load(TopKCoverageScore())

    metrics = scorer.apply(
        smiles_predictions["sampled_smiles"],
        smiles_predictions["target_smiles"],
    )

    assert round(metrics["coverage_top_1"], 4) == 0.3333
    assert round(metrics["coverage_top_3"], 4) == 0.4444
    assert round(metrics["coverage_top_5"], 4) == 0.6
    assert "coverage_top_10" not in metrics

    scorer.load(TopKCoverageScore(n_predictions=10))

    metrics = scorer.apply(
        smiles_predictions["sampled_smiles"],
        smiles_predictions["target_smiles"],
    )

    assert round(metrics["coverage_top_1"], 4) == 0.3333
    assert round(metrics["coverage_top_3"], 4) == 0.4444
    assert round(metrics["coverage_top_5"], 4) == 0.6
    assert round(metrics["coverage_top_10"], 4) == 0.3


def test_accuracy_similarity(smiles_predictions):
    scorer = ScoreCollection()
    scorer.load(TanimotoSimilarityScore(statistics="all"))
    scorer.load(TopKAccuracyScore())

    metrics = scorer.apply(
        smiles_predictions["sampled_smiles"],
        smiles_predictions["target_smiles"],
    )

    assert any(sim == 1.0 for sim in metrics["top1_tanimoto_similarity"][0])
    assert round(metrics["accuracy_top_1"], 4) == 0.3333
    assert round(metrics["accuracy_top_3"], 4) == 0.6667
    assert round(metrics["accuracy_top_5"], 4) == 1.0


def test_accuracy_similarity_grouped(smiles_predictions_grouped):
    scorer = ScoreCollection()
    scorer.load(TanimotoSimilarityScore(statistics="all"))
    scorer.load(TopKAccuracyScore(top_ks=[1, 2, 5]))

    metrics = scorer.apply(
        smiles_predictions_grouped["sampled_smiles"],
        smiles_predictions_grouped["target_smiles"],
    )

    assert any(sim == 1.0 for sim in metrics["top1_tanimoto_similarity"][0])
    assert round(metrics["accuracy_top_1"], 4) == 0.3333
    assert round(metrics["accuracy_top_2"], 4) == 0.6667
    assert round(metrics["accuracy_top_5"], 4) == 1.0


def test_similarity_wrong_input():
    scorer = ScoreCollection()
    with pytest.raises(ValueError, match="'statistics' should be either 'mean', 'median' or 'all'"):
        scorer.load(TanimotoSimilarityScore(statistics="var"))


def test_accuracy_numeric(torch_predictions_class):
    scorer = ScoreCollection()
    scorer.load(TopKAccuracyScore())

    metrics = scorer.apply(
        torch_predictions_class["predictions"], torch_predictions_class["ground_truth"], is_canonical=True
    )

    assert round(metrics["accuracy_top_1"], 4) == 0.25
    assert round(metrics["accuracy_top_3"], 4) == 0.75


def test_mae(torch_predictions_float):
    scorer = ScoreCollection()
    scorer.load(MeanAbsoluteError())

    metrics = scorer.apply(
        torch_predictions_float["predictions"],
        torch_predictions_float["ground_truth"],
    )

    assert round(metrics["mae"], 2) == 0.02


def test_r2(torch_predictions_float):
    scorer = ScoreCollection()
    scorer.load(R2Score())

    metrics = scorer.apply(
        torch_predictions_float["predictions"],
        torch_predictions_float["ground_truth"],
    )

    assert round(metrics["r2"], 2) == 0.75


def test_balanced_accuracy(torch_predictions_binary_class):
    scorer = ScoreCollection()
    scorer.load(BalancedAccuracyScore())

    metrics = scorer.apply(
        torch_predictions_binary_class["predictions"],
        torch_predictions_binary_class["ground_truth"],
    )

    assert round(metrics["balanced_accuracy"], 2) == 0.5


def test_average_precision(torch_predictions_binary_class):
    scorer = ScoreCollection()
    scorer.load(AveragePrecisionScore())

    metrics = scorer.apply(
        torch_predictions_binary_class["predictions"],
        torch_predictions_binary_class["ground_truth"],
    )

    assert round(metrics["average_precision"], 2) == 0.5


def test_recall(torch_predictions_binary_class):
    scorer = ScoreCollection()
    scorer.load(RecallScore())

    metrics = scorer.apply(
        torch_predictions_binary_class["predictions"],
        torch_predictions_binary_class["ground_truth"],
    )

    assert round(metrics["recall"], 2) == 0.5
