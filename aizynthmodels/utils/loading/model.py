from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from aizynthmodels.utils.loading import load_item, unravel_list_dict

if TYPE_CHECKING:
    from typing import Union

    from aizynthmodels.utils.type_utils import StrDict


def build_model(
    model_config: Union[DictConfig, StrDict], module_name: str, model_hyperparams: Union[StrDict, DictConfig], mode: str
) -> pl.LightningModule:
    """
    Load a model from config. The keys are 'type' and 'arguments'. The value for 'type'
    is the name of model class to load, while 'arguments' contain input arguments to the model.
    If a model is not defined in the module defined by 'module_name'
    (e.g. aizynthmodels.chemformer.models), the module name can be explicitly stated
    in the config e.g. ``mypackage.models.AwesomeModel``.

    :param model_config: A dictionary defining the model class ('type') and
        input parameters ('arguments').
    :param module_name: The base module where the class is found.
    :param extra_kwargs: Additional input args passed to the class.
    """

    if isinstance(model_hyperparams, DictConfig):
        model_hyperparams = OmegaConf.to_container(model_hyperparams, resolve=True)

    if model_config.get("arguments"):
        model_hyperparams.update(unravel_list_dict(model_config["arguments"]))

    cls, kwargs, _ = load_item(model_config["type"], module_name, model_hyperparams)

    if model_hyperparams.get("ckpt_path"):
        model = initialize_from_ckpt(cls, model_hyperparams, mode)
    else:
        model = random_initialization(cls, model_hyperparams)

    config_str = f" with configuration '{kwargs}'"
    logging.info(f"Loaded model: '{model_config['type']}'{config_str}")
    return model


def random_initialization(model_cls: pl.LightningModule, model_hyperparams: StrDict) -> pl.LightningModule:
    """Constructing a model with randomly initialized weights."""
    model = model_cls(config=model_hyperparams)
    return model


def initialize_from_ckpt(model_cls: pl.LightningModule, model_hyperparams: StrDict, mode: str) -> pl.LightningModule:
    """Constructing a model with weights from a ckpt-file."""
    model = model_cls.load_from_checkpoint(model_hyperparams["ckpt_path"], config=model_hyperparams)
    if mode == "eval":
        model.eval()
    return model
