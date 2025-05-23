"""
Config for running Chemformer encoder and obtain latent space.
"""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.chemformer.core import CoreConfig


@dataclass
class Encode(CoreConfig):
    n_chunks: int = 1  # Number of chunks to divide the data into (for parallel inference)
    i_chunk: int = 0  # The idx of the current chunk
    dataset_part: str = "test"  # Which dataset split to run inference on. [full", "train", "val", "test"]

    output_encodings: str = "encodings.csv"


cs = ConfigStore.instance()
cs.store(name="encode", node=Encode)
