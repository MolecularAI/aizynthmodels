import copy
import logging
from typing import Any, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig

from aizynthmodels.model_interface import ModelInterface
from aizynthmodels.route_distances.data.datamodule import __name__ as data_module
from aizynthmodels.route_distances.models import LstmDistanceModel
from aizynthmodels.utils.loading import build_datamodule
from aizynthmodels.utils.type_utils import StrDict


class RouteDistanceModel(ModelInterface):
    def __init__(self, config: DictConfig) -> None:

        self.config = config

        if self.config.model_path:
            self.model = LstmDistanceModel.load_from_checkpoint(self.config.model_path)
            if self.config.mode == "eval":
                self.model.eval()
        else:
            self.model = LstmDistanceModel(config=self.config.model_hyperparams)

        self.set_datamodule(datamodule_config=self.config.get("datamodule"))

        super().__init__(self.config)
        self.model.scores = self.scores

    def on_device(self, batch: StrDict) -> StrDict:
        """
        Move data in "batch" to the current model device.

        :param batch: batch input data to model.
        :return: batch data on current device.
        """
        device_batch = copy.copy(batch)
        for key in batch.keys():
            device_batch[key] = self._on_device_partial_data(device_batch[key])
        return device_batch

    def _on_device_partial_data(self, partial_data: Union[torch.Tensor, StrDict]) -> Union[torch.Tensor, StrDict]:
        if isinstance(partial_data, torch.Tensor):
            return partial_data.to(self.device)

        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in partial_data.items()
        }
        return device_batch

    def set_datamodule(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        datamodule_config: Optional[Union[ListConfig, DictConfig]] = None,
    ) -> None:
        """
        Create a new datamodule by either supplying a datamodule (created elsewhere) or
        a pre-defined datamodule type as input.

        :param datamodule: pytorchlightning datamodule
        :param datamodule_config: The config of datamodule to build if no
            pytorchlightning datamodule is given as input.
        If no inputs are given, the datamodule will be specified using the config.
        """
        if datamodule is None and datamodule_config is not None:
            self.datamodule = build_datamodule(datamodule_config, data_module, {"dataset_path": self.config.data_path})
        elif datamodule is None:
            logging.info("Did not initialize datamodule.")
            return
        else:
            self.datamodule = datamodule

        setup_kwargs = {}
        if str(self.datamodule) == "TreeListDataModule":
            setup_kwargs = {"fp_size": self.model.hparams.fp_size}
        self.datamodule.setup(**setup_kwargs)
        return

    def _predict_batch(self, batch: Any) -> Tuple[List, ...]:
        predicted_distances = self.model(batch["tree"]).detach().cpu().numpy()
        predictions = {"predictions": predicted_distances}
        if "ted" in batch:
            predictions["ground_truth"] = batch["ted"]
        return predictions
