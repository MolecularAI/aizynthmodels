""" Module containing helper routines for the DataModules """

from omegaconf import DictConfig

from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer, build_masker
from aizynthmodels.chemformer.utils.defaults import DEFAULT_MAX_SEQ_LEN
from aizynthmodels.utils.type_utils import StrDict


def construct_datamodule_kwargs(config: DictConfig, tokenizer: ChemformerTokenizer) -> StrDict:
    """
    Returns a dictionary with kwargs which are general to the BaseDataModule.
    These are specified as single parameters in the config file
    """

    masker = build_masker(config, tokenizer)

    reverse = config.task == "backward_prediction"
    kwargs = {
        "reverse": reverse,
        "max_seq_len": config.get("max_seq_len", DEFAULT_MAX_SEQ_LEN),
        "tokenizer": tokenizer,
        "masker": masker,
        "augment_prob": config.get("augmentation_probability"),
        "dataset_path": config.data_path,
        "batch_size": config.batch_size,
        "add_sep_token": config.get("add_sep_token", False),
        "i_chunk": config.get("i_chunk", 0),
        "n_chunks": config.get("n_chunks", 1),
    }
    return kwargs
