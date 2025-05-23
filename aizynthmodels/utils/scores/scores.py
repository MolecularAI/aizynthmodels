from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchmetrics
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score

from aizynthmodels.utils.smiles import canonicalize_smiles, inchi_key
from aizynthmodels.utils.type_utils import PredictionType, TargetType


class BaseScore:
    """
    Base scoring class.
    """

    scorer_name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        return

    def __call__(self, predictions: PredictionType, ground_truth: TargetType = None) -> Dict[str, float]:
        return self._score_predictions(predictions, ground_truth)

    def __repr__(self):
        return self.scorer_name

    def _score_predictions(self, predictions: PredictionType, ground_truth: TargetType = None) -> Dict[str, Any]:
        """Scoring function which should be implemented in each new Score class."""
        raise NotImplementedError("self._score_predictions() needs to be implemented for every scoring class.")

    def _is_grouped(self, predictions: PredictionType) -> bool:
        if isinstance(predictions, torch.Tensor):
            return False
        elif not (isinstance(predictions[0][0], np.ndarray) or isinstance(predictions[0][0], list)):
            return False
        return True

    def _ungroup(self, predictions: PredictionType) -> PredictionType:
        """If the predictions are grouped according to ranks, they will be unravelled."""
        if not self._is_grouped(predictions):
            return predictions

        predictions_ungrouped = []
        for top_k_predictions in predictions:
            top_k_unravelled = [pred for predictions in top_k_predictions for pred in predictions]
            predictions_ungrouped.append(top_k_unravelled)
        return predictions_ungrouped


class FractionInvalidScore(BaseScore):
    """
    Scoring using fraction of invalid of all or top-1 SMILES.
    """

    scorer_name = "fraction_invalid"

    def __init__(self, only_top1: bool = False):
        """
        :param only_top1: If True, will only compute fraction of invalid top-1 SMILES,
                otherwise fraction invalid is over all generated SMILES.
        """
        super().__init__()
        self.only_top1 = only_top1

    def _score_predictions(self, sampled_smiles: PredictionType, target_smiles: TargetType = None) -> Dict[str, float]:
        """Computing fraction of invalid SMILES."""

        sampled_smiles = self._ungroup(sampled_smiles)

        if self.only_top1:
            is_valid = [
                (bool(Chem.MolFromSmiles(top_k_smiles[0])) if len(top_k_smiles) > 0 else False)
                for top_k_smiles in sampled_smiles
            ]
        else:
            is_valid = []
            for top_k_smiles in sampled_smiles:
                for smiles in top_k_smiles:
                    is_valid.append(bool(Chem.MolFromSmiles(smiles)))

        fraction_invalid = 1 - (sum(is_valid) / len(is_valid))
        return {self.scorer_name: fraction_invalid}

    def _ungroup(self, predictions: PredictionType) -> PredictionType:
        """If the predictions are grouped according to ranks, they will be unravelled."""
        if not self._is_grouped(predictions):
            return predictions

        predictions_ungrouped = []
        for top_k_predictions in predictions:
            top_k_unravelled = [predictions[0] for predictions in top_k_predictions]
            predictions_ungrouped.append(top_k_unravelled)
        return predictions_ungrouped


class FractionUniqueScore(BaseScore):
    """
    Scoring using the fraction of uniquely sampled SMILES among the top-N sampled SMILES.
    """

    scorer_name = "fraction_unique"

    def __init__(self, canonicalized: bool = False):
        """
        :param canonicalized: whether the sampled_smiles and target_smiles are
            been canonicalized.
        :param grouped: whether the sampled_smiles are grouped according to same rank
            probability.
        """
        super().__init__()
        self._canonicalized = canonicalized

    def _score_predictions(
        self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Computing fraction of unique top-N SMILES."""

        sampled_smiles = self._ungroup(sampled_smiles)

        n_samples = len(sampled_smiles)
        n_predictions = len(sampled_smiles[0])

        n_unique_total = 0
        for top_k in sampled_smiles:
            if not self._canonicalized:
                top_k = [inchi_key(smiles) for smiles in top_k if Chem.MolFromSmiles(smiles)]
            else:
                top_k = [smiles for smiles in top_k if Chem.MolFromSmiles(smiles)]
            n_unique = len(set(top_k))
            n_unique_total += n_unique
        fraction_unique = n_unique_total / (n_predictions * n_samples)
        return {self.scorer_name: fraction_unique}


class MeanAbsoluteError(BaseScore):
    scorer_name = "mae"

    def __init__(self):
        super().__init__()
        self._mae = None

    def _score_predictions(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        if not self._mae:
            self._mae = torchmetrics.MeanAbsoluteError().to(predictions.device)

        mae = float(self._mae(predictions, ground_truth).detach().cpu().numpy())
        return {self.scorer_name: mae}


class R2Score(BaseScore):
    scorer_name = "r2"

    def __init__(self):
        super().__init__()
        self._r2 = None

    def _score_predictions(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        if not self._r2:
            self._r2 = torchmetrics.R2Score().to(predictions.device)

        r2 = float(self._r2(predictions, ground_truth).detach().cpu().numpy())
        return {self.scorer_name: r2}


class AveragePrecisionScore(BaseScore):
    scorer_name = "average_precision"

    def __init__(self):
        super().__init__()

    def _score_predictions(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        precision = average_precision_score(predictions.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
        return {self.scorer_name: precision}


class RecallScore(BaseScore):
    scorer_name = "recall"

    def __init__(self):
        super().__init__()

    def _score_predictions(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        recall = recall_score(predictions.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
        return {self.scorer_name: recall}


class BalancedAccuracyScore(BaseScore):
    scorer_name = "balanced_accuracy"

    def __init__(self):
        super().__init__()

    def _score_predictions(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        accuracy = balanced_accuracy_score(predictions.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
        return {self.scorer_name: accuracy}


class BinaryAccuracyScore(BaseScore):
    scorer_name = "binary_accuracy"

    def __init__(self):
        super().__init__()
        self._accuracy = None

    def _score_predictions(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        if not self._accuracy:
            self._accuracy = torchmetrics.Accuracy(task="binary").to(predictions.device)

        accuracy = float(self._accuracy(predictions, ground_truth).detach().cpu().numpy())
        return {self.scorer_name: accuracy}


class TanimotoSimilarityScore(BaseScore):
    """
    Scoring using the Tanomoto similarity of the top-1 sampled SMILES and the target
    SMILES.
    """

    scorer_name = "top1_tanimoto_similarity"

    def __init__(self, statistics="mean"):
        """
        :param return_strategy: ["mean", "median", "all"], returns the average similarity or
            all similarities.
        """
        super().__init__()

        if statistics not in ["mean", "median", "all"]:
            raise ValueError(f"'statistics' should be either 'mean', 'median' or 'all'," f" not {statistics}")
        self._statistics = statistics

        self._stat_fcn = {"mean": np.mean, "median": np.median}

    def _get_statistics(self, similarities: np.ndarray) -> float:
        if self._statistics == "all":
            return [similarities]
        similarities = similarities[~np.isnan(similarities)]
        return self._stat_fcn[self._statistics](similarities)

    def _score_predictions(
        self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute similarities of ECPF4 fingerprints of target and top-1 sampled molecules.
        """
        sampled_smiles = self._ungroup(sampled_smiles)

        target_molecules = [Chem.MolFromSmiles(smiles) for smiles in target_smiles]

        sampled_molecules = [
            Chem.MolFromSmiles(smiles_list[0]) if len(smiles_list) > 0 else None for smiles_list in sampled_smiles
        ]

        n_samples = len(target_molecules)

        similarities = np.nan * np.ones(n_samples)
        counter = 0
        for sampled_mol, target_mol in zip(sampled_molecules, target_molecules):
            if not sampled_mol or not target_mol:
                counter += 1
                continue

            fp1 = AllChem.GetMorganFingerprint(sampled_mol, 2)
            fp2 = AllChem.GetMorganFingerprint(target_mol, 2)

            similarities[counter] = DataStructs.TanimotoSimilarity(fp1, fp2)  # Tanimoto similarity = Jaccard similarity
            counter += 1

        return {self.scorer_name: self._get_statistics(similarities)}


class TopKAccuracyScore(BaseScore):
    scorer_name = "top_k_accuracy"

    def __init__(
        self,
        top_ks: Union[List, np.ndarray] = np.array([1, 3, 5, 10, 20, 30, 40, 50]),
        canonicalized: bool = False,
    ):
        """
        :param top_ks: a list of top-Ks to compute accuracy for.
        :param canonicalized: whether the sampled_smiles and target_smiles are
                been canonicalized.
        :param grouped: whether the sampled_smiles are grouped according to same rank
            probability.
        """
        super().__init__()
        if isinstance(top_ks, list):
            top_ks = np.array(top_ks)
        self._top_ks: np.ndarray = top_ks
        self._canonicalized = canonicalized
        self._grouped = False

    def _is_in_set(self, predictions: PredictionType, ground_truth: TargetType, k: int) -> np.ndarray:
        if not self._canonicalized:
            ground_truth = [canonicalize_smiles(smiles) for smiles in ground_truth]

            if not self._grouped:
                predictions = [[canonicalize_smiles(smiles) for smiles in smiles_list] for smiles_list in predictions]
            else:
                predictions = [
                    [[canonicalize_smiles(smiles) for smiles in grouped_smiles] for grouped_smiles in top_k_predictions]
                    for top_k_predictions in predictions
                ]

        if not self._grouped:
            is_in_set = [
                tgt_smi in sampled_smi[0:k] if len(sampled_smi[0:k]) > 0 else False
                for sampled_smi, tgt_smi in zip(predictions, ground_truth)
            ]
        else:
            is_in_set = []
            for sampled_smi, tgt_smi in zip(predictions, ground_truth):
                is_in_set.append(False)
                if len(sampled_smi[0:k]) > 0:
                    for grouped_smiles in sampled_smi[0:k]:
                        if tgt_smi in grouped_smiles:
                            is_in_set[-1] = True
                            break
        return is_in_set

    def _score_predictions(self, predictions: PredictionType, ground_truth: TargetType) -> Dict[str, float]:

        self._grouped = self._is_grouped(predictions)

        n_predictions = np.max(np.array([1, np.max(np.asarray([len(prediction) for prediction in predictions]))]))
        top_ks = self._top_ks[self._top_ks <= n_predictions]

        columns = []
        is_in_set = np.zeros((len(predictions), len(top_ks)), dtype=bool)
        for i_k, k in enumerate(top_ks):
            columns.append(f"accuracy_top_{k}")
            is_in_set[:, i_k] = self._is_in_set(predictions, ground_truth, k)

        is_in_set = np.cumsum(is_in_set, axis=1)
        top_n_accuracy = np.mean(is_in_set > 0, axis=0)

        scores = {col: accuracy for col, accuracy in zip(columns, top_n_accuracy)}
        return scores


class TopKCoverageScore(BaseScore):
    scorer_name = "top_k_coverage"

    def __init__(
        self,
        top_ks: Union[List, np.ndarray] = np.array([1, 3, 5, 10, 20, 30, 40, 50]),
        n_predictions: Optional[int] = None,
        canonicalized: bool = False,
    ):
        """
        Compute fraction of predictions which represent the correct target SMILES.
        Typically used for round-trip accuracy.

        :param top_ks: a list of top-Ks to compute coverage for.
        :param n_predictions: number of predictions (could be prior to e.g. uniqueifying).
        :param canonicalized: whether the sampled_smiles and target_smiles are
                been canonicalized.
        """
        super().__init__()
        if isinstance(top_ks, list):
            top_ks = np.array(top_ks)
        self._top_ks = top_ks
        self._canonicalized = canonicalized
        self._n_predictions = n_predictions

    def _check_accuracy(self, predictions: PredictionType, ground_truth: TargetType, n_predictions: int) -> np.ndarray:
        if not self._canonicalized:
            ground_truth = [canonicalize_smiles(smiles) for smiles in ground_truth]

            predictions = [
                [canonicalize_smiles(smiles) for smiles in top_k_predictions] for top_k_predictions in predictions
            ]

        batch_size = len(predictions)
        is_accurate = np.zeros((batch_size, n_predictions), dtype=bool)
        sample_idx = 0
        for sampled_smi, tgt_smi in zip(predictions, ground_truth):
            is_accurate[sample_idx, :] = [
                tgt_smi == sampled_smi[idx] if len(sampled_smi) > idx else False for idx in range(n_predictions)
            ]
            sample_idx += 1
        return is_accurate

    def _score_predictions(self, predictions: PredictionType, ground_truth: TargetType) -> Dict[str, float]:

        self._grouped = self._is_grouped(predictions)

        n_predictions = (
            self._n_predictions
            if self._n_predictions is not None
            else np.max(np.array([1, np.max(np.asarray([len(prediction) for prediction in predictions]))]))
        )
        top_ks = self._top_ks[self._top_ks <= n_predictions]

        columns = []
        is_accurate = self._check_accuracy(predictions, ground_truth, n_predictions)
        coverages = np.zeros((len(predictions), len(top_ks)), dtype=float)
        for i_k, k in enumerate(top_ks):
            columns.append(f"coverage_top_{k}")
            coverages[:, i_k] = np.mean(is_accurate[:, 0:k], axis=1)

        top_k_coverage = np.mean(coverages, axis=0)
        scores = {col: accuracy for col, accuracy in zip(columns, top_k_coverage)}
        return scores
