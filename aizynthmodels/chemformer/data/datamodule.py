""" Module containing the default datamodules"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from rdkit import Chem
from torch.nn.functional import one_hot

from aizynthmodels.chemformer.data.base import BaseDataModule, MoleculeListDataModule, ReactionListDataModule
from aizynthmodels.chemformer.data.encoder import BatchEncoder
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
from aizynthmodels.utils.tokenizer import SMILESAugmenter


class ChemblDataModule(MoleculeListDataModule):
    """
    DataModule for Chembl dataset.

    The molecules and the lengths of the sequences
    are loaded from a pickled DataFrame
    """

    def __repr__(self):
        return "ChemblDataModule"

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path)
        self._all_data = {
            "molecules": df["molecules"].tolist(),
            "lengths": df["lengths"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        smiles_batch = [{"smiles": Chem.MolToSmiles(item["molecules"])} for item in batch]
        return super()._transform_batch(smiles_batch, train)


class ZincDataModule(MoleculeListDataModule):
    """
    DataModule for Zinc dataset.

    The molecules are read as SMILES from a number of
    csv files.
    """

    def __repr__(self):
        return "ZincDataModule"

    def _load_all_data(self) -> None:
        path = Path(self.dataset_path)
        if path.is_dir():
            dfs = [pd.read_csv(filename) for filename in path.iterdir()]
            df = pd.concat(dfs, ignore_index=True, copy=False)
        else:
            df = pd.read_csv(path)
        self._all_data = {"smiles": df["smiles"].tolist()}
        self._set_split_indices_from_dataframe(df)


class SynthesisDataModule(ReactionListDataModule):
    """
    DataModule for forward and backard synthesis prediction.

    The reactions are read from a tab seperated DataFrame .csv file.
    Expects the dataset to contain SMILES in two seperate columns named "reactants" and "products".
    The dataset must also contain a columns named "set" with values of "train", "val" and "test".
    validation column can be named "val", "valid" or "validation".

    Supports both loading data from file, and in-memory prediction.

    All rows that are not test or validation, are assumed to be training samples.
    """

    def __init__(
        self,
        reactants: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        augmentation_strategy: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._augment_strategy = augmentation_strategy
        self._in_memory = False
        if reactants is not None and products is not None:
            self._in_memory = True
            logging.info("Using in-memory datamodule.")
            self._all_data = {"reactants": reactants, "products": products}

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [item["reactants"] for item in batch]
        products = [item["products"] for item in batch]
        if train:
            if self._augment_strategy == "reactants" or self._augment_strategy == "all":
                reactants = self._batch_augmenter(reactants)
            if self._augment_strategy == "products" or self._augment_strategy == "all":
                products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        if self._in_memory:
            return

        if self.dataset_path.endswith(".csv"):
            df = pd.read_csv(self.dataset_path, sep="\t").reset_index()
            self._all_data = {
                "reactants": df["reactants"].tolist(),
                "products": df["products"].tolist(),
            }
            self._set_split_indices_from_dataframe(df)
        else:
            super()._load_all_data()


class ClassificationDataModule(BaseDataModule):
    """
    DataModule for forward and backard synthesis prediction.
    The reactions are read from a tab seperated DataFrame .csv file.
    Expects the dataset to contain SMILES and corresponding class labels in two
    seperate columns, default named "smiles" and "label".
    The dataset must also contain a columns named "set" with values of "train", "val" and "test".
    validation column can be named "val", "valid" or "validation".
    Supports both loading data from file, and in-memory prediction.
    All rows that are not test or validation, are assumed to be training samples.

    :param dataset_path: the path to the dataset
    :param tokenizer: the tokenizer to use
    :param batch_size: the batch size to use
    :param max_seq_len: the maximum allowed sequence length
    :param input_column: column name corresponding to the input SMILE
    :param label_column: name of the column containing the labels
    :param target_smiles_column: name of the column with the target SMILES (if applicable)
    :param smiles: list of input SMILES, if using in-memory mode
    :param labels: list of input labels, if using in-memory mode
    :param augment_prob: the probability of augmenting the sequences in training
    """

    def __init__(
        self,
        dataset_path: Optional[str],
        tokenizer: ChemformerTokenizer,
        batch_size: int,
        max_seq_len: int,
        input_column: str = "smiles",
        label_column: str = "label",
        target_smiles_column: Optional[str] = None,
        smiles: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        augment_prob: int = 0.0,
        **kwargs
    ):

        kwargs.update(
            {"dataset_path": dataset_path, "tokenizer": tokenizer, "batch_size": batch_size, "max_seq_len": max_seq_len}
        )
        super().__init__(**kwargs)
        self._in_memory = False
        self.input_column = input_column
        self.label_column = label_column
        self.target_smiles_column = target_smiles_column
        self.num_classes = None

        if smiles and labels:
            self._in_memory = True
            logging.info("Using in-memory datamodule.")
            self._all_data = {"input_smiles": smiles, "label": labels}

        self._batch_augmenter = SMILESAugmenter(augment_prob=augment_prob)
        self._encoder = BatchEncoder(tokenizer=self.tokenizer, masker=None, max_seq_len=self.max_seq_len)

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[int], torch.Tensor, Optional[List[int]]]:
        inputs = [item["input_smiles"] for item in batch]
        labels = [item["label"] for item in batch]
        class_indicator = torch.stack([item["class_indicator"] for item in batch])

        if batch[0].get("target_smiles"):
            targets = [item["target_smiles"] for item in batch]
        else:
            targets = None

        if train and self._batch_augmenter.augment_prob > 0.0:
            inputs = self._batch_augmenter(inputs)

        return inputs, labels, class_indicator, targets

    def _load_all_data(self) -> None:
        if self._in_memory:
            self.num_classes = max(set(self._all_data["label"])) + 1
            class_indicator = self._one_hot_encode_labels(torch.tensor(self._all_data["label"]).to(int))
            self._all_data.update({"class_indicator": class_indicator})
            return None

        df = pd.read_csv(self.dataset_path, sep="\t").reset_index()
        self._all_data = {
            "input_smiles": df[self.input_column].tolist(),
            "label": df[self.label_column].astype(int).tolist(),
        }
        if self.target_smiles_column:
            self._all_data["target_smiles"] = df[self.target_smiles_column].tolist()

        self.num_classes = max(set(self._all_data["label"])) + 1
        class_indicator = self._one_hot_encode_labels(torch.from_numpy(df[self.label_column].values.astype(int)))
        self._all_data.update({"class_indicator": class_indicator})
        self._set_split_indices_from_dataframe(df)

    def _one_hot_encode_labels(self, labels: torch.Tensor) -> torch.Tensor:
        class_indicator = one_hot(labels, self.num_classes).to(torch.float64)
        return class_indicator

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], Optional[List[str]]]:
        encoder_smiles, labels, class_indicator, target_smiles = self._get_sequences(batch, train)
        encoder_ids, encoder_mask = self._encoder(encoder_smiles, add_sep_token=False)
        return encoder_ids, encoder_mask, labels, class_indicator, encoder_smiles, target_smiles

    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        (encoder_ids, encoder_mask, labels, class_indicator, input_smiles, target_smiles) = self._transform_batch(
            batch, train
        )

        batch = {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "label": labels,
            "class_indicator": class_indicator,
            "input_smiles": input_smiles,
        }

        if target_smiles is not None:
            batch["target_smiles"] = (target_smiles,)

        return batch
