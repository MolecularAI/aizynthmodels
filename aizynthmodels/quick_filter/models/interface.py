""" Module containing the interface to the quick-filter model
"""

import logging
from typing import Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig, OmegaConf

from aizynthmodels.model_interface import ModelInterface
from aizynthmodels.quick_filter import ClassificationModel
from aizynthmodels.quick_filter.data import __name__ as data_module
from aizynthmodels.utils.loading import build_datamodule
from aizynthmodels.utils.type_utils import StrDict


class QuickFilter(ModelInterface):
    """
    Interface to training and evaluating a quick-filter model

    :param config: the configuration for the model
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = config.device

        super().__init__(config)

        self.set_datamodule(datamodule_config=self.config.get("datamodule"))
        self._setup_model()
        self.model.scores = self.scores

    def set_datamodule(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        datamodule_config: Optional[Union[ListConfig, str]] = None,
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
            self.datamodule = build_datamodule(datamodule_config, data_module, {"batch_size": self.config.batch_size})
        elif datamodule is None:
            logging.info("Did not initialize datamodule.")
            return
        else:
            self.datamodule = datamodule
        self.datamodule.setup()

    def _setup_model(self):
        if self.mode.startswith("train"):
            model_hyperparams = OmegaConf.to_container(self.config.model_hyperparams, resolve=True)
            model_hyperparams["num_features"] = self.datamodule.train_dataset.prod_input_matrix.shape[1]
            model_hyperparams["threshold"] = self.config.threshold

            self.model = ClassificationModel(config=model_hyperparams)

            if self.config.model_path is not None:
                if self.config.model_path.endswith(".onnx"):
                    self.model.load_weights_from_onnx(self.config.onnx_model)
                else:
                    self.model = ClassificationModel.load_from_checkpoint(self.config.model_path)
        else:
            self.model = ClassificationModel.load_from_checkpoint(self.config.model_path)
            self.model.eval()

    def to_onnx(self, filename: str) -> None:
        """
        Convert to ONNX format using a sample from the dataloader

        :param filename: the path to the ONNX model
        """
        data = iter(self.datamodule.test_dataloader())
        first_batch = next(data)
        input_sample = (first_batch["product_input"][:1, :], first_batch["reaction_input"][:1, :])
        input_names = [name for name, _ in self.model.named_parameters()]
        self.model.to_onnx(
            filename,
            input_sample,
            export_params=True,
            input_names=input_names,
            dynamic_axes={input_names[0]: [0], input_names[2]: [0]},
        )

    def _prediction_setup(self, n_predictions: Optional[int] = None) -> None:
        self.n_predictions: int = n_predictions or self.config.n_predictions
        self.model.n_predictions = self.n_predictions

    def _predict_batch(self, batch: StrDict) -> StrDict:
        """
        Generates predictions for an input batch from the datamodule
        """
        return self.model.sample_predictions(batch, self.n_predictions)
