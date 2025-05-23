from pathlib import Path

import pandas as pd

from aizynthmodels.utils.writing import predictions_to_file


def test_writing(tmpdir):

    ground_truth = ["pred1_1", "pred_2_3", "pred3_1"]

    predictions = [
        ["pred1_1", "pred1_2", "pred1_3"],
        ["pred2_1", "pred2_2", "pred2_3"],
        ["pred3_1", "pred3_2", "pred3_3"],
    ]

    probabilities = [
        [0.6, 0.3, 0.1],
        [0.7, 0.15, 0.15],
        [0.75, 0.15, 0.1],
    ]

    df_expected = pd.DataFrame(
        {
            "prediction_1": ["pred1_1", "pred2_1", "pred3_1"],
            "prediction_2": ["pred1_2", "pred2_2", "pred3_2"],
            "prediction_3": ["pred1_3", "pred2_3", "pred3_3"],
            "probability_1": [0.6, 0.7, 0.75],
            "probability_2": [0.3, 0.15, 0.15],
            "probability_3": [0.1, 0.15, 0.1],
        }
    )

    filename = f"{tmpdir}/output_predictions.csv"

    predictions_to_file(filename, predictions, probabilities)

    assert Path(filename).exists()

    df = pd.read_csv(filename, sep="\t")
    pd.testing.assert_frame_equal(df, df_expected)

    predictions_to_file(filename, predictions, probabilities, ground_truth)
    df = pd.read_csv(filename, sep="\t")
    df_expected["ground_truth"] = ground_truth

    pd.testing.assert_frame_equal(
        df,
        df_expected[
            [
                "ground_truth",
                "prediction_1",
                "prediction_2",
                "prediction_3",
                "probability_1",
                "probability_2",
                "probability_3",
            ]
        ],
    )
