from typing import List, Tuple

import numpy as np

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.chemformer.data import ClassificationDataModule, SynthesisDataModule
from aizynthmodels.chemformer.utils.defaults import DEFAULT_MAX_SEQ_LEN
from aizynthmodels.utils.tokenizer import SMILESAugmenter


def calculate_llhs(chemformer: Chemformer, reactants: List[str], products: List[str]) -> List[float]:
    """
    Calculate log-likelihood of reactant/product pairs.
    """
    datamodule = SynthesisDataModule(
        reactants=reactants,
        products=products,
        dataset_path="",
        tokenizer=chemformer.tokenizer,
        batch_size=chemformer.config.batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        augment_prob=False,
        reverse=chemformer.config.task == "backward_prediction",
    )

    datamodule.setup()
    llhs = chemformer.log_likelihood(dataloader=datamodule.full_dataloader())
    return llhs


def estimate_compound_llhs(
    chemformer: Chemformer,
    reactants: List[str],
    products: List[str],
    n_augments: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use SMILES augmentation to generate multiple SMILES representations of a
    compound and compute the log-likelihood of each SMILES.
    Returns the maximum log-likelihood.
    """
    augmenter = SMILESAugmenter()
    base_log_likelihoods = np.array(calculate_llhs(chemformer, reactants, products))

    all_llhs = []
    all_llhs.append(base_log_likelihoods[:, np.newaxis])

    for _ in range(n_augments - 1):
        if chemformer.config.task == "forward_prediction":
            this_products = augmenter(products)
            this_reactants = reactants
        else:
            this_products = products
            this_reactants = augmenter(reactants)

        aug_log_likelihoods = np.array(calculate_llhs(chemformer, this_reactants, this_products))

        all_llhs.append(aug_log_likelihoods[:, np.newaxis])

    best_log_likelihoods = np.concatenate(all_llhs, axis=1)
    best_log_likelihoods = np.max(best_log_likelihoods, axis=1)

    return best_log_likelihoods


def get_predictions(
    chemformer: Chemformer, smiles_list: List[str], n_predictions: int = 10
) -> Tuple[List[List[str]], List[List[float]], List[str]]:
    """
    Predict with Chemformer on input smiles_list.
    """
    # Setting both reactants and products to smiles_list since we do
    # not have the "ground truth" data.
    datamodule = SynthesisDataModule(
        reactants=smiles_list,
        products=smiles_list,
        tokenizer=chemformer.tokenizer,
        batch_size=chemformer.config.batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        dataset_path="",
    )
    datamodule.setup()

    output = chemformer.predict(dataloader=datamodule.full_dataloader(), n_predictions=n_predictions)
    return output["predictions"], output["log_likelihoods"], output["ground_truth"]


def get_classifier_predictions(
    chemformer: Chemformer, smiles_list: List[str], n_predictions: int = 10
) -> Tuple[List[List[str]], List[List[float]], List[str]]:
    """
    Predict with Chemformer classifier on input smiles_list.
    """
    datamodule = ClassificationDataModule(
        smiles=smiles_list,
        labels=[0] * len(smiles_list),  # placeholder arg since we do not have the ground-truth
        tokenizer=chemformer.tokenizer,
        batch_size=chemformer.config.batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        dataset_path="",
    )
    datamodule.setup()

    chemformer.model.n_predictions = n_predictions

    output = chemformer.predict(dataloader=datamodule.full_dataloader())
    return output["predictions"], output["log_likelihoods"]
