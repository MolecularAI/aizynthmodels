""" Module for preparing model tree-LSTM training input """

import logging
import pickle

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from aizynthmodels.route_distances.utils.features import preprocess_reaction_tree

# flake8: noqa: F401
from aizynthmodels.utils.configs.route_distances import prepare_data
from aizynthmodels.utils.hydra import custom_config


def _similarity(idx1, idx2, labels):
    if len(labels) == 0 or (labels[idx1] == labels[idx2]):
        return 1, 1
    return -1, 0


@hydra.main(version_base=None, config_name="prepare_data")
@custom_config
def main(config: DictConfig) -> None:
    offset = 0
    tree_list = []
    pairs = []
    for filename in tqdm(config.datapath, desc="# of files processed: "):
        data = pd.read_hdf(filename, "table")

        for trees, distances, labels in zip(
            tqdm(data.trees.values, leave=False, desc="# of targets processed"),
            data.distance_matrix.values,
            data.cluster_labels.values,
        ):
            np_distances = np.asarray(distances)
            for i, tree1 in enumerate(trees):
                tree_list.append(preprocess_reaction_tree(tree1, config.fp_size))
                for j, _ in enumerate(trees):
                    if j < i and config.use_reduced:
                        continue
                    loss_target, pair_similarity = _similarity(i, j, labels)
                    pairs.append(
                        (
                            i + offset,
                            j + offset,
                            np_distances[i, j],
                            pair_similarity,
                            loss_target,
                        )
                    )
            offset += len(trees)

    logging.info(f"Preprocessed {len(tree_list)} trees in {len(pairs)} pairs")

    with open(config.output, "wb") as fileobj:
        pickle.dump(
            {
                "trees": tree_list,
                "pairs": pairs,
            },
            fileobj,
        )


if __name__ == "__main__":
    main()
