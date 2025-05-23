""" Contains extensions and utilities to hydra
"""

from __future__ import annotations

import os
from dataclasses import is_dataclass
from functools import wraps
from typing import Any, Callable

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def instantiate_config(config_path: str) -> DictConfig:
    """Instantiate a hydra config from a given path."""
    config_dir = os.path.dirname(config_path) or os.getcwd()
    config_dir = os.path.abspath(config_dir)
    config_name = os.path.basename(config_path)

    GlobalHydra.instance().clear()
    config = DictConfig({})
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        config = hydra.compose(config_name=config_name)
    return config


def load_config(config_override_path: str, config_default_obj: Any) -> DictConfig:
    """
    Load a config file which overrides default parameter values.

    :param config_override_path: Path to the config file that should be loaded.
    :param config_default_path: Path to the config file with default values.
    :return: The loaded hydra config.
    """
    if isinstance(config_default_obj, str):
        config_default = instantiate_config(config_default_obj)
    elif is_dataclass(config_default_obj):
        config_default = OmegaConf.structured(config_default_obj)
    else:
        raise ValueError("config_default_obj should be either a path to a .yaml file (str), or a dataclass.")
    config_override = instantiate_config(config_override_path)
    return OmegaConf.merge(config_default, config_override)


def custom_config(func: Callable) -> Callable:
    """
    Decorator that can be used after the hydra.main decorator to load
    and override config with settings in a custom file

    Example:
        import hydra

        from aizynthmodels.utils.hydra import custom_config

        @hydra.main(version_base=None, config_path="config", config_name="base-config")
        @custom_config
        def main(config):
            print(config)


        python my_app.py +config=custom.yaml

    Everything that is provided by the file pointed to by the config-override be used
    to overload the base config and the overrides provided on the command-line
    """

    @wraps(func)
    def inner_decorator(config: DictConfig):
        if "config" in config:
            config_path = config["config"]
            config_override = instantiate_config(config_path)
            config = OmegaConf.merge(config, config_override)
        return func(config)

    return inner_decorator
