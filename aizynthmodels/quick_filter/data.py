""" Module containing classes for loading and generating data for model training and inference """

import multiprocessing
from typing import Dict

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from aizynthmodels.template_based.data import SmilesBasedDataModule as BaseSmilesBasedDataModule
from aizynthmodels.utils.smiles import seq_rxn_smiles_to_fingerprint, seq_smiles_to_fingerprint


class InMemoryLabeledDataset(Dataset):
    """Represent an in-memory set of labeled data and its input features"""

    def __init__(
        self,
        prod_features: sparse.csr_matrix,
        reaction_features: sparse.csr_matrix,
        labels: np.ndarray,
    ) -> None:
        self.prod_input_matrix = prod_features
        self.reaction_input_matrix = reaction_features
        self.label_vector = labels

    def __len__(self) -> int:
        return len(self.label_vector)

    def __getitem__(self, item: int) -> Dict[str, np.array]:
        return {
            "product_input": self.prod_input_matrix[item, :].toarray().flatten().astype(np.float32),
            "reaction_input": self.reaction_input_matrix[item, :].toarray().flatten().astype(np.float32),
            "label": np.float32(self.label_vector[item]),
        }


class PrecomputedDataModule(LightningDataModule):
    """
    Represent a PyTorch Lightning datamodule for loading and
    collecting data for model training using pre-computed featurizations and labels
    :params files_prefix: the prefix of the data files
    :params inputs_prod_postfix: the postfix for the  product features data files
    :params inputs_rxn_postfix: the postfix for the  reaction features data files
    :params labels_postfix: the postfix for the labels data files
    :params batch_size: the batch size
    :params shuffle: if True, will shuffle the training data
    """

    def __init__(
        self,
        files_prefix: str,
        inputs_prod_postfix: str,
        inputs_rxn_postfix: str,
        labels_postfix: str,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self._files_prefix = files_prefix
        self._inputs_prod_postfix = inputs_prod_postfix
        self._inputs_rxn_postfix = inputs_rxn_postfix
        self._labels_postfix = labels_postfix
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = multiprocessing.cpu_count()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None) -> None:
        self.train_dataset = InMemoryLabeledDataset(
            self._load_mat_data("training", "inputs_prod"),
            self._load_mat_data("training", "inputs_rxn"),
            self._load_np_data("training"),
        )
        self.val_dataset = InMemoryLabeledDataset(
            self._load_mat_data("validation", "inputs_prod"),
            self._load_mat_data("validation", "inputs_rxn"),
            self._load_np_data("validation"),
        )
        self.test_dataset = InMemoryLabeledDataset(
            self._load_mat_data("testing", "inputs_prod"),
            self._load_mat_data("testing", "inputs_rxn"),
            labels=self._load_np_data("testing"),
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

    def _load_mat_data(self, setname: str, matname: str) -> sparse.csr_matrix:
        if matname == "inputs_prod":
            filename = f"{self._files_prefix}_{setname}_{self._inputs_prod_postfix}"
        else:
            filename = f"{self._files_prefix}_{setname}_{self._inputs_rxn_postfix}"
        matrix = sparse.load_npz(filename)
        assert isinstance(matrix, sparse.csr_matrix)
        return matrix

    def _load_np_data(self, setname: str) -> np.ndarray:
        filename = f"{self._files_prefix}_{setname}_{self._labels_postfix}"
        arr = np.load(filename)["arr_0"]
        assert isinstance(arr, np.ndarray)
        return arr


class SmilesBasedDataModule(BaseSmilesBasedDataModule):

    def setup(self) -> None:
        df = pd.read_csv(self._dataset_path, sep="\t").reset_index()

        val_idxs = df.query("set in ['val','valid','validation']").index.tolist()
        train_idxs = df.query("set in ['train','Train']").index.tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        inputs_prod = np.apply_along_axis(
            seq_smiles_to_fingerprint,
            0,
            [df["products"].to_numpy()],
            fp_length=self._fp_size,
            fp_radius=self._fp_radius,
            chirality=self._fp_chirality,
        )
        inputs_prod = sparse.lil_matrix(inputs_prod.T).tocsr()

        inputs_rxn = np.apply_along_axis(
            seq_rxn_smiles_to_fingerprint,
            0,
            [df["products"].to_numpy(), df["reactants"].to_numpy()],
            fp_length=self._fp_size,
            fp_radius=self._fp_radius,
            chirality=self._fp_chirality,
        )
        inputs_rxn = sparse.lil_matrix(inputs_rxn.T).tocsr()

        self.full_dataset = InMemoryLabeledDataset(
            prod_features=inputs_prod, reaction_features=inputs_rxn, labels=np.asarray([np.nan] * inputs_prod.shape[0])
        )
        self.train_dataset = InMemoryLabeledDataset(
            prod_features=inputs_prod[train_idxs, :],
            reaction_features=inputs_rxn[train_idxs, :],
            labels=np.asarray([np.nan] * len(train_idxs)),
        )
        self.val_dataset = InMemoryLabeledDataset(
            prod_features=inputs_prod[val_idxs, :],
            reaction_features=inputs_rxn[val_idxs, :],
            labels=np.asarray([np.nan] * len(val_idxs)),
        )
        self.test_dataset = InMemoryLabeledDataset(
            prod_features=inputs_prod[test_idxs, :],
            reaction_features=inputs_rxn[test_idxs, :],
            labels=np.asarray([np.nan] * len(test_idxs)),
        )
