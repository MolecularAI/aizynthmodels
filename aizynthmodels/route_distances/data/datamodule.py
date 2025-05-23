""" Module containing classes for loading and generating data from model training """

import json
import logging
import multiprocessing
import pickle
import random
from typing import Callable, List, Optional, Set, Tuple, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from aizynthmodels.route_distances.utils.data import collate_batch, collate_trees
from aizynthmodels.route_distances.utils.features import preprocess_reaction_tree
from aizynthmodels.utils.type_utils import RouteList

_PairType = Tuple[Union[int, float], ...]


class InMemoryTreeDataset(Dataset):
    """Represent an in-memory set of trees, and pairwise distances"""

    def __init__(self, pairs, trees):
        self.trees = trees
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        tree_index1, tree_index2, *target_values = self.pairs[item]
        item = {
            "tree1": self.trees[tree_index1],
            "tree2": self.trees[tree_index2],
            "ted": target_values[0],
        }
        return item


class TreeDataModule(LightningDataModule):
    """Represent a PyTorch Lightning datamodule for loading and collecting data for model training"""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 128,
        split_part: float = 0.1,
        split_seed: int = 1984,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._split_part = split_part
        self._split_seed = split_seed
        self._shuffle = shuffle
        self._num_workers = multiprocessing.cpu_count()

        self._all_pairs = []
        self._all_trees = []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        with open(self._dataset_path, "rb") as fileobj:
            raw_data = pickle.load(fileobj)
        self._all_pairs = raw_data["pairs"]
        self._all_trees = raw_data["trees"]
        self._split_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
        )

    def full_dataloader(self):
        """Returns the DataLoader for the full dataset"""

        dataset_len = len(self._all_pairs)
        pair_segments = self._make_segments()
        full_dataset = self._make_dataset(pair_segments, set(range(len(pair_segments))), dataset_len)

        return DataLoader(
            full_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
        )

    def _collate(self, data) -> Callable:
        return collate_batch(data)

    def _make_dataset(
        self,
        segments: List[List[_PairType]],
        indices: Set[int],
        sample_size: int,
    ) -> InMemoryTreeDataset:
        segment_ids = list(indices)
        random.shuffle(segment_ids)
        taken = []
        for segment_id in segment_ids:
            taken.extend(segments[segment_id])
            indices -= {segment_id}
            if len(taken) >= sample_size:
                break
        return InMemoryTreeDataset(taken, self._all_trees)

    def _make_segments(self) -> List[List[_PairType]]:
        segments = []
        segment = []
        seen = set()
        for pair in self._all_pairs:
            idx1, idx2, *_ = pair
            if idx1 == idx2 and idx2 not in seen and segment:
                segments.append(segment)
                segment = []
            segment.append(pair)
            seen.add(idx1)
            seen.add(idx2)
        segments.append(segment)
        return segments

    def _split_data(self) -> None:
        """
        Split the data into training, validation and test set using a
        segmented approach. First the pairs are split into non-overlapping
        pairs, corresponding to different sets of target molecules.
        Second a dataset is built up by adding all pairs from a segment until
        a sufficiently large dataset has been created.
        """
        random.seed(self._split_seed)

        dataset_len = len(self._all_pairs)
        val_len = round(dataset_len * self._split_part)
        train_len = dataset_len - 2 * val_len

        pair_segments = self._make_segments()
        if len(pair_segments) < 3:
            raise ValueError(
                f"Could only make {len(pair_segments)} segments from the pairs. Unable to split the dataset"
            )

        indices = set(range(len(pair_segments)))
        self.train_dataset = self._make_dataset(pair_segments, indices, train_len)
        self.val_dataset = self._make_dataset(pair_segments, indices, val_len)
        self.test_dataset = self._make_dataset(pair_segments, indices, val_len)

        logging.info("=== Data split ===")
        logging.info(
            f"Training dataset: {len(self.train_dataset)} ({len(self.train_dataset)/len(self._all_pairs)*100:.2f}%)"
        )
        logging.info(
            f"Validation dataset: {len(self.val_dataset)} ({len(self.val_dataset) / len(self._all_pairs) * 100:.2f}%)"
        )
        logging.info(
            f"Test dataset: {len(self.test_dataset)} ({len(self.test_dataset) / len(self._all_pairs) * 100:.2f}%)"
        )


class TreeListDataModule(TreeDataModule):
    """
    Datamodule for loading and collecting a list of routes for pairwise
    distance prediction
    """

    def __init__(self, dataset_path: Optional[str] = None, route_list: Optional[RouteList] = None) -> None:
        """
        :param dataset_path: Path to .json with list of AiZF routes. Overrides 'route_list'.
        :param route_list: List of AiZF routes (in dictionary format). For in-memory predictions.
        """
        super().__init__(dataset_path=dataset_path)
        self._route_list = route_list

    def setup(self, fp_size: int = 2048, stage: str = None) -> None:
        if self._dataset_path:
            with open(self._dataset_path, "r") as fileobj:
                self._route_list = json.load(fileobj)
        self._all_trees = [preprocess_reaction_tree(route, fp_size) for route in self._route_list]

    def _collate(self, data) -> Callable:
        return {"tree": collate_trees(data)}

    def full_dataloader(self):
        """Returns the DataLoader for the full dataset"""

        return DataLoader(
            self._all_trees,
            batch_size=len(self._all_trees),
            collate_fn=self._collate,
            num_workers=self._num_workers,
        )
