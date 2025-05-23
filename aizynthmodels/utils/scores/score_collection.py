from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import ListConfig

from aizynthmodels.utils.loading import load_item, unravel_list_dict
from aizynthmodels.utils.scores import BaseScore
from aizynthmodels.utils.scores import __name__ as score_module

if TYPE_CHECKING:
    from typing import Dict, List, Optional

    from aizynthmodels.utils.type_utils import PredictionType, StrDict, TargetType


class ScoreCollection:
    """
    Store score objects for the chemformer model.

    The scores can be obtained by name

    .. code-block::

        scores = ScoreCollection()
        score = scores['TopKAccuracy']
    """

    _collection_name = "scores"

    def __init__(self) -> None:
        self._logger = logging.Logger("score-collection")
        self._items: Dict[str, BaseScore] = {}

    def names(self) -> List[str]:
        """Returns a list of the names of each loaded item."""
        return list(self._items.keys())

    def objects(self) -> List[BaseScore]:
        """Return a list of all the loaded items"""
        return list(self._items.values())

    def load(self, score: BaseScore) -> None:  # type: ignore
        """
        Add a pre-initialized score object to the collection

        :param score: the item to add
        """
        if not isinstance(score, BaseScore):
            raise ValueError(
                "Only objects of classes inherited from " "aizynthmodels.utils.scores.BaseScore can be added"
            )
        self._items[str(score)] = score
        self._logger.info(f"Loaded score: {str(score)}")

    def load_from_config(self, scores_config: ListConfig) -> None:
        """
        Load one or several scores from a configuration dictionary

        The keys are the name of score class. If a score is not
        defined in the ``aizynthmodels.utils.scores`` module, the module
        name can be appended, e.g. ``mypackage.scoring.AwesomeScore``.

        The values of the configuration is passed directly to the score
        class along with the ``config`` parameter.

        :param scores_config: Config of scores
        """
        for item in scores_config:
            cls, kwargs, config_str = load_item(item, score_module)

            # Convert possible ListConfig arguments to lists
            for key, val in kwargs.items():
                if isinstance(val, ListConfig):
                    kwargs[key] = list(val)

            obj = cls(**kwargs)
            self._items[repr(obj)] = obj
            logging.info(f"Loaded score: '{repr(obj)}'{config_str}")

    def apply(
        self, predictions: PredictionType, ground_truth: TargetType = None, is_canonical: Optional[bool] = None
    ) -> StrDict:
        """
        Apply all scores in collection to the given sampled and target SMILES.

        :param predictions: predictions, e.g. top-N SMILES sampled by a model, such as Chemformer.
        :param ground_truth: ground truth labels or SMILES.
        :param is_canonical: Whether the predictions are canonicalized
            (True - numeric data and canonical SMILES, False - Noncanonical SMILES)
        :raises: ValueError if the number of predictions and ground-truth samples are different.
        :return: A dictionary with all the scores.
        """
        n_pred_samples = len(predictions)
        n_target_samples = len(ground_truth)
        if n_pred_samples != n_target_samples:
            raise ValueError(
                f"The number of predictions and targets must be the same, "
                f"got {n_pred_samples} and {n_target_samples}"
            )

        scores = []
        for score in self._items.values():
            if is_canonical is not None and hasattr(score, "_canonicalized"):
                setattr(score, "_canonicalized", is_canonical)

            scores.append(score(predictions, ground_truth))

        return unravel_list_dict(scores)
