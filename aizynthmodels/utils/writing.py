import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd


def predictions_to_file(
    filename: str,
    predictions: Union[List[List[str]], List[List[int]]],
    ranking_scores: List[List[float]],
    ground_truth: Optional[List[str]] = None,
    prediction_col: str = "prediction",
    ranking_metric_col: str = "probability",
) -> None:
    """
    Writing predictions to file, either .json or .csv, depending on filename ending.

    :param filename: name of the output file
    :param predictions: list of top-K predictions
    :param ranking_scores: list of top-K ranking metric, e.g. probabilities, log-likelihoods, etc.
    :param ground_truth: ground truth output
    :param prediction_col: name of column(s) in output dataframe with predictions
    :param ranking_metric_col: name of column(s) in output dataframe with ranking metrics
    """

    n_samples = len(predictions)
    n_predictions = max([len(top_k_preds) for top_k_preds in predictions])
    output_preds = [[""] * n_samples for _ in range(n_predictions)]
    output_probs = np.zeros((n_predictions, n_samples))

    for idx, (top_k_preds, top_k_probs) in enumerate(zip(predictions, ranking_scores)):
        for prediction_idx, (prediction, prob) in enumerate(zip(top_k_preds, top_k_probs)):
            output_preds[prediction_idx][idx] = prediction
            output_probs[prediction_idx, idx] = prob

    data = {"ground_truth": ground_truth} if ground_truth else {}
    for prediction_idx in range(n_predictions):
        data[f"{prediction_col}_{prediction_idx + 1}"] = output_preds[prediction_idx]

    for prediction_idx in range(n_predictions):
        data[f"{ranking_metric_col}_{prediction_idx + 1}"] = output_probs[prediction_idx, :]

    df = pd.DataFrame(data)

    if filename.endswith(".json"):
        df.to_json(filename, orient="table")
    else:
        df.to_csv(filename, sep="\t", index=False)
    logging.info(f"Predictions saved to: {filename}")
