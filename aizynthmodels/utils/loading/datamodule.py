import logging
from typing import Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.utils.loading import load_item, unravel_list_dict
from aizynthmodels.utils.type_utils import StrDict


def build_datamodule(
    datamodule_config: Union[DictConfig, StrDict], module_name: str, extra_kwargs: Optional[StrDict] = None
) -> pl.LightningDataModule:
    """
    Loads a datamodule from config.
    The keys are 'type' and 'arguments'. The value for 'type' is the name of datamodule
    class, while 'arguments' contain input arguments to the datamodule.
    If a datamodule is not defined in the module defined by 'module_name'
    (e.g. aizynthmodels.chemformer.data.datamodules), the module name can be prepended,
    in the config e.g. ``mypackage.datamodules.AwesomeDataModule``.

    :param datamodule_config: A dictionary defining the datamodule class ('type') and
        input parameters ('arguments').
    :param module_name: The base module where the class is found.
    :param extra_kwargs: Additional input args passed to the class.
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    if datamodule_config.get("arguments"):
        extra_kwargs.update(unravel_list_dict(datamodule_config["arguments"]))

    cls, kwargs, _ = load_item(datamodule_config["type"], module_name, extra_kwargs)
    obj = cls(**kwargs)
    config_str = f" with configuration '{kwargs}'"
    logging.info(f"Loaded datamodule: '{datamodule_config['type']}'{config_str}")
    return obj
