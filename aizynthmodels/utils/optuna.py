"""module containing an objective class and utility function for Optuna hyperparameter optimization"""

import copy
import logging
from typing import Any

import optuna
import pandas as pd
from omegaconf import DictConfig

from aizynthmodels.utils.type_utils import StrDict


class OptunaObjective:
    """
    Representation of an objective function for Optuna.
    :param config: Input hydra config file for the model.
    :param model_interface: The model interface with the model for which hyperparameters
        will be optimized.
    """

    def __init__(self, config: DictConfig, model_interface: Any) -> None:
        self._config = config
        self._opt_config = self._config.opt_hyperparams
        self._model_interface = model_interface

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Run training using one hyperparameter configuration, randomly selected given
        the input config.

        :param trial: Optuna trial object
        :return: Score / objective function value of the used configuration
        """

        config = self._config
        model_hyperparams = copy.copy(config.model_hyperparams)

        for hparam_dict in self._opt_config:
            for hparam_name, hparam_opt_config in hparam_dict.items():
                if hparam_name.count(".") == 1:
                    base_key, hparam_key = hparam_name.split(".")
                    model_hyperparams[base_key][hparam_key] = self._generate_hparam(
                        trial, hparam_name, hparam_opt_config
                    )
                else:
                    model_hyperparams[hparam_name] = self._generate_hparam(trial, hparam_name, hparam_opt_config)

        config.model_hyperparams = model_hyperparams
        try:
            model = self._model_interface(config)
        except Exception as err:
            logging.info(f"Model hyperparams: {model_hyperparams}")
            raise err
        model.fit()
        return model.trainer.callback_metrics[config.optuna.objective].item()

    def _generate_hparam(self, trial: optuna.trial.Trial, hparam_name: str, config: StrDict) -> Any:
        """
        Update hyperparameter values with trial using the input config.

        :param trial: Optuna trial object
        :param hparam_name: Name of the hyperparameter
        :param config: Hyperparameter configuration used to generate candidate value
        :return: Generated hyperparameter value
        """
        hparam_type = config["type"]
        if hparam_type == "float":
            return trial.suggest_float(hparam_name, config["values"][0], config["values"][1], **config.get("args", {}))
        elif hparam_type == "int":
            if not config.get("choice"):
                return trial.suggest_int(
                    hparam_name, config["values"][0], config["values"][1], **config.get("args", {})
                )
            else:
                # Generate an index and return real hyperparam value from "choice"
                return config["choice"][
                    trial.suggest_int(
                        f"{hparam_name}_idx", config["values"][0], config["values"][1], **config.get("args", {})
                    )
                ]
        elif hparam_type == "categorical":
            return trial.suggest_categorical(hparam_name, config["values"], **config.get("args", {}))
        else:
            raise ValueError(
                f"Hyperparameter type can be one of ['float', 'int', 'categorical'], but got {hparam_type}"
            )


def optimize_model(config: DictConfig, model_interface_cls: Any) -> pd.DataFrame:
    """
    Creating an objective function for Optuna and running hyperparameter optimization.
    :param config: Input hydra config file for the model and optuna arguments.
    :param model_interface_cls: The model interface class with the model for which hyperparameters
        will be optimized.
    """
    storage_name = "sqlite:///{}.db".format(config.optuna.study_name)

    if config.optuna.mode == "grid_search" and config.search_space is None:
        raise ValueError(
            "The parameter 'search_space' has to be specified to run " "hyperparameter optimization with grid search."
        )

    kwargs = (
        {
            "sampler": optuna.samplers.GridSampler(
                {search_space["name"]: search_space["values"] for search_space in config.search_space}
            )
        }
        if config.optuna.mode == "grid_search"
        else {}
    )

    study = optuna.create_study(
        direction=config.optuna.direction,
        study_name=config.optuna.study_name,
        storage=storage_name,
        load_if_exists=config.optuna.load_if_exists,
        **kwargs,
    )

    study.optimize(OptunaObjective(config, model_interface_cls), n_trials=config.optuna.n_trials)

    best_trial = study.best_trial
    best_params = study.best_params
    best_params["version_number"] = best_trial.number
    best_params["objective_value"] = best_trial.values[0]

    for key, val in best_params.items():
        best_params[key] = [val]

    return pd.DataFrame(best_params)
