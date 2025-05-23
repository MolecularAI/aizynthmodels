""" Module containing classes for loading and generating data for model training and inference """

import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from rxnutils.chem.utils import remove_atom_mapping, split_rsmi
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from aizynthmodels.utils.smiles import seq_smiles_to_fingerprint


class InMemoryLabeledDataset(Dataset):
    """Represent an in-memory set of labeled data and its input features"""

    def __init__(
        self,
        features: sparse.csr_matrix,
        labels: sparse.csr_matrix,
    ) -> None:
        self.input_matrix = features
        self.label_matrix = labels

    def __len__(self) -> int:
        return self.label_matrix.shape[0]

    def __getitem__(self, item: int) -> Dict[str, np.array]:
        return {
            "input": self.input_matrix[item, :].toarray().flatten().astype(np.float32),
            "label": self.label_matrix[item, :].toarray().flatten().astype(np.float32),
        }


class InMemoryLabeledReactionDataset(InMemoryLabeledDataset):
    """Represent an in-memory set of labeled reaction data and its input features"""

    def __init__(
        self,
        features: sparse.csr_matrix,
        labels: Optional[sparse.csr_matrix],
        reactants: List[str],
        products: List[str],
    ):
        super().__init__(features, labels if labels is not None else sparse.csr_matrix((features.shape[0], 0)))
        self.reactants = reactants
        self.products = products

    def __getitem__(self, item: int) -> Dict[str, Any]:
        dict_ = super().__getitem__(item)
        dict_["product"] = self.products[item]
        dict_["reactant"] = self.reactants[item]
        return dict_


class PrecomputedDataModule(LightningDataModule):
    """
    Represent a PyTorch Lightning datamodule for loading and
    collecting data for model training using pre-computed featurizations and labels
    :params files_prefix: the prefix of the data files
    :params inputs_postfix: the postfix for the features data files
    :params labels_postfix: the postfix for the labels data files
    :params batch_size: the batch size
    :params shuffle: if True, will shuffle the training data
    """

    def __init__(
        self,
        files_prefix: str,
        inputs_postfix: str,
        labels_postfix: str,
        reactions_postfix: Optional[str] = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self._files_prefix = files_prefix
        self._inputs_postfix = inputs_postfix
        self._labels_postfix = labels_postfix
        self._reactions_postfix = reactions_postfix
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = multiprocessing.cpu_count()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self) -> None:
        self.train_dataset = InMemoryLabeledDataset(
            self._load_data("training", "inputs"),
            self._load_data("training", "labels"),
        )
        self.val_dataset = InMemoryLabeledDataset(
            self._load_data("validation", "inputs"),
            self._load_data("validation", "labels"),
        )

        if self._reactions_postfix:
            reactants, products = self._load_reaction_data("testing")

            self.test_dataset = InMemoryLabeledReactionDataset(
                self._load_data("testing", "inputs"),
                labels=self._load_data("testing", "labels"),
                reactants=reactants,
                products=products,
            )
        else:
            self.test_dataset = InMemoryLabeledDataset(
                self._load_data("testing", "inputs"),
                labels=self._load_data("testing", "labels"),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def _load_data(self, setname: str, matname: str) -> sparse.csr_matrix:
        if matname == "inputs":
            filename = f"{self._files_prefix}_{setname}_{self._inputs_postfix}"
        else:
            filename = f"{self._files_prefix}_{setname}_{self._labels_postfix}"
        matrix = sparse.load_npz(filename)
        assert isinstance(matrix, sparse.csr_matrix)
        return matrix

    def _load_reaction_data(self, setname: str) -> Tuple[List[str], List[str]]:
        filename = f"{self._files_prefix}_{setname}_{self._reactions_postfix}"
        data = pd.read_csv(filename, sep="\t")
        reactants = []
        products = []
        for reaction_smiles in data.reaction_smiles.values:
            reactant_smiles, _, product_smiles = split_rsmi(reaction_smiles)
            reactants.append(remove_atom_mapping(reactant_smiles, sanitize=False))
            products.append(remove_atom_mapping(product_smiles, sanitize=False))
        return reactants, products


class SmilesBasedDataModule(LightningDataModule):
    """
    Represent a PyTorch Lightning datamodule for loading and
    collecting data for model training based on a reaction dataset
    The dataset read from a CSV file contains columns with products,
    reactants and set (train, val, or test). The featurization of the
    products is done on-the-fly when the `setup` method is called.
    The hyper-parameters of the featurization has to be given to the
    class on instantiation.
    :params dataset_path: the path of the CSV file with the data
    :params fingerprint_size: the length of the fingerprint vector
    :params fingerprint_radius: the radius of the fingerprints
    :params chirality: if True, will include chirality information in the fingerprint
    :params batch_size: the batch size
    :params shuffle: if True, will shuffle the training data
    """

    def __init__(
        self,
        dataset_path: str,
        fingerprint_size: int,
        fingerprint_radius: int,
        chirality: bool,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._fp_size = fingerprint_size
        self._fp_radius = fingerprint_radius
        self._fp_chirality = chirality
        self._shuffle = shuffle
        self._num_workers = multiprocessing.cpu_count()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None

    def setup(self) -> None:
        df = pd.read_csv(self._dataset_path, sep="\t").reset_index()

        val_idxs = df.query("set in ['val','valid','validation']").index.tolist()
        train_idxs = df.query("set in ['train','Train']").index.tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        inputs = np.apply_along_axis(
            seq_smiles_to_fingerprint,
            0,
            [df["products"].to_numpy()],
            fp_length=self._fp_size,
            fp_radius=self._fp_radius,
            chirality=self._fp_chirality,
        )
        inputs = sparse.lil_matrix(inputs.T).tocsr()

        self.full_dataset = InMemoryLabeledReactionDataset(
            features=inputs, labels=None, reactants=df["reactants"].to_list(), products=df["products"].to_list()
        )
        self.train_dataset = InMemoryLabeledReactionDataset(
            features=inputs[train_idxs, :],
            labels=None,
            reactants=df["reactants"].iloc[train_idxs].to_list(),
            products=df["products"].iloc[train_idxs].to_list(),
        )
        self.val_dataset = InMemoryLabeledReactionDataset(
            features=inputs[val_idxs, :],
            labels=None,
            reactants=df["reactants"].iloc[val_idxs].to_list(),
            products=df["products"].iloc[val_idxs].to_list(),
        )
        self.test_dataset = InMemoryLabeledReactionDataset(
            features=inputs[test_idxs, :],
            labels=None,
            reactants=df["reactants"].iloc[test_idxs].to_list(),
            products=df["products"].iloc[test_idxs].to_list(),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def full_dataloader(self) -> DataLoader:
        return DataLoader(
            self.full_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )
