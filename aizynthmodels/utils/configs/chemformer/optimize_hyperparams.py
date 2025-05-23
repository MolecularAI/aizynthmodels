from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from aizynthmodels.utils.configs.chemformer.fine_tune import FineTune
from aizynthmodels.utils.type_utils import StrDict


@dataclass
class OptimizeHyperparams(FineTune):
    """
    Shares the basic arguments from FineTune and includes two additional:

    * optuna: Configuration for optuna Trial object which runs the optimization
    * opt_hyperparams:
        List of hyperparameters to optimize. Includes these keys:
        - type: ["int", "float", "categorical"]
        - values: List with max/min range for 'int' and 'float', or list of
                ints for 'categorical'.
        - choice: Paired with 'int' if 'categorical' cannot handle datatype
                (e.g. list with layer configs). Then, "values" should cover the indices
                in the list
        - args: Other keyword arguments to be passed to the methods, e.g. 'step' or 'log'.

    Example .yaml config for setting some hyperparameters to optimize with optuna
    (num_layers, learning_rate and num_hidden_nodes):
    ```
    opt_hyperparams:
        - num_layers:
            type: int
            values: [2, 6]
            args:
                step: 2
        - optimizer.learning_rate:
            type: float
            values: [1e-5, 5e-3]
            args:
                log: True
        - num_hidden_nodes:
            type: int
            values: [0, 4]
            choice: [[], [32], [256], [256, 128], [128, 64]]   # List of Lists is not supported by categorical
            args:
                step: 1
    ```
    """

    optuna: StrDict = field(
        default_factory=lambda: {
            "study_name": "hyperparam_opt",
            "n_trials": 50,
            "objective": "validation_loss",
            "direction": "minimize",
            "mode": "random",
            "output_hyperparams": "chemformer/optuna/best_hyperparams.csv",
            "load_if_exists": True,
        }
    )

    opt_hyperparams: List[StrDict] = field(
        default_factory=lambda: [
            {"num_layers": {"type": "int", "values": [2, 6], "args": {"step": 2}}},
            {
                "d_feedforward": {
                    "type": "categorical",
                    "values": [512, 1024, 2048],
                }
            },
            {
                "d_model": {
                    "type": "categorical",
                    "values": [256, 512, 1024],
                }
            },
            {"dropout": {"type": "float", "values": [0.1, 0.8], "args": {"step": 0.1}}},
            {"optimizer.learning_rate": {"type": "float", "values": [1e-5, 5e-3], "args": {"log": True}}},
        ]
    )

    search_space: Optional[List[Any]] = None  # Only needed when mode="grid_search"

    output_directory: str = "chemformer/optuna"


cs = ConfigStore.instance()
cs.store(name="optimize_hyperparams", node=OptimizeHyperparams)
