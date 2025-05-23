import logging
import os
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from aizynthmodels.chemformer.data.datamodule import __name__ as data_module
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer
from aizynthmodels.chemformer.models import __name__ as models_module
from aizynthmodels.chemformer.models.base_transformer import BaseTransformer
from aizynthmodels.chemformer.sampler import SMILESSampler
from aizynthmodels.chemformer.utils.data import construct_datamodule_kwargs
from aizynthmodels.chemformer.utils.models import set_model_hyperparams
from aizynthmodels.model_interface import ModelInterface
from aizynthmodels.utils.loading import build_datamodule, build_model
from aizynthmodels.utils.trainer import calc_train_steps
from aizynthmodels.utils.type_utils import StrDict


class Chemformer(ModelInterface):
    """
    Class for building (synthesis) Chemformer model, fine-tuning seq-seq model,
    and predicting/scoring model.

    :param config: OmegaConf config loaded by hydra. Contains the input args of the model,
        including vocabulary, model checkpoint, beam size, etc.
    """

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """
        The config includes the following arguments:
            # Trainer args
            seed: 1
            batch_size: 128
            n_devices (int): Number of devices to use.
            i_chunk: 0              # For inference
            n_chunks: 1             # For inference
            limit_val_batches: 1.0  # For training
            n_buckets: 12           # For training
            n_nodes: 1              # For training
            acc_batches: 1          # For training
            accelerator: null       # For training

            # Data args
            data_path (str): path to data used for training or inference
            backward_predictions (str): path to sampled smiles (for round-trip inference)
            dataset_part (str): Which dataset split to run inference on. ["full", "train", "val", "test"]
            vocabulary_path (str): path to bart_vocabulary.
            task (str): the model task ["forward_prediction", "backward_prediction"]

            # Model args
            model_path (Optional[str]): Path to model weights.
            model_type (str): the model type ["bart", "bart_sp"]
            n_predictions (int): Number of predictions to sample with the sampler.
            sample_unique (bool): Return the unique predictions.
                If None => return all unique solutions.
            mode(str): Whether to train the model ("train") or use
                model for evaluations ("eval").
            device (str): Which device to run model and beam search on ("cuda" / "cpu").
            resume_training (bool): Whether to continue training from the supplied
                .ckpt file.

            learning_rate (float): the learning rate (for training/fine-tuning)
            weight_decay (float): the weight decay (for training/fine-tuning)

            # Transformer model parameters
            model_hyperparams:
                d_model (int): 512
                n_layers (int): 6
                n_heads (int): 8
                d_feedforward (int): 2048

            callbacks: list of Callbacks
            datamodule: the DataModule to use

            # Inference args
            scorers: list of Scores to evaluate sampled smiles against target smiles
            output_score_data: null
            output_sampled_smiles: null
        """
        self.config = config
        self.tokenizer = ChemformerTokenizer(filename=config.vocabulary_path)
        self.set_datamodule(datamodule_config=config.get("datamodule"))

        super().__init__(config)

        logging.info("Vocabulary_size: " + str(len(self.tokenizer)))
        self.train_steps = None
        if self.mode.startswith("train"):
            self.train_steps = calc_train_steps(config, self.datamodule)
            logging.info(f"Train steps: {self.train_steps}")

        n_classes = getattr(self.datamodule, "num_classes", None)
        model_hyperparams = set_model_hyperparams(self.config, self.tokenizer, self.train_steps, n_classes)
        self.model: BaseTransformer = build_model(self.config.model, models_module, model_hyperparams, self.config.mode)

        self.sampler = None
        if self.config.get("sampler"):
            self.sampler = SMILESSampler(
                self.tokenizer,
                self.config.model_hyperparams.max_seq_len,
                device=self.device,
                sample_unique=self.config.get("sample_unique", False),
                sampler_node=self.config.sampler,
                batch_size=self.config.batch_size,
            )
            self.model.sampler = self.sampler

        self.model.n_predictions = self.config.n_predictions
        self.model.scores = self.scores

    def set_datamodule(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        datamodule_config: Optional[Union[DictConfig, StrDict]] = None,
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
            self.datamodule = build_datamodule(
                datamodule_config,
                data_module,
                construct_datamodule_kwargs(self.config, self.tokenizer),
            )
        elif datamodule is None:
            logging.info("Did not initialize datamodule.")
            self.datamodule = None
            return
        else:
            self.datamodule = datamodule

        self.datamodule.setup()
        n_cpus = len(os.sched_getaffinity(0))
        if self.config.n_devices > 0:
            n_workers = n_cpus // self.config.n_devices
        else:
            n_workers = n_cpus
        self.datamodule._num_workers = n_workers
        logging.info(f"Using {str(n_workers)} workers for data module.")
        return

    def encode(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[torch.Tensor]:
        """
        Compute memory from transformer inputs.

        :param dataset: (Which part of the dataset to use (["train", "val", "test",
                "full"]).)
        :param dataloader: (If None -> dataloader
                will be retrieved from self.datamodule)
        :return: Tranformer embedding/memory
        """

        self.model.to(self.device)
        self.model.eval()

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        X_encoded = []
        for b_idx, batch in enumerate(dataloader):
            batch = self.on_device(batch)
            with torch.no_grad():
                batch_encoded = self.model.encode(batch).permute(
                    1, 0, 2
                )  # Return on shape [n_samples, n_tokens, max_seq_length]
            
            if self.device != "cpu":
                batch_encoded = batch_encoded.detach().cpu()
            else:
                batch_encoded = batch_encoded.detach()
            
            X_encoded.extend(batch_encoded)
        return X_encoded

    def decode(
        self,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output token probabilities from a given decoder input

        :param memory_input: tensor from encoded input of shape (src_len,
                batch_size, d_model)
        :param memory_pad_mask: bool tensor of memory padding mask of shape
                (src_len, batch_size)
        :param decoder_input: tensor of decoder token_ids of shape (tgt_len,
                batch_size)
        :return: Output probabilities
        """
        self.model.to(self.device)
        self.model.eval()

        batch_input = {
            "memory_input": memory,
            "memory_pad_mask": memory_pad_mask.permute(1, 0),
            "decoder_input": decoder_input.permute(1, 0),
            "decoder_pad_mask": torch.zeros_like(decoder_input, dtype=bool).permute(1, 0),
        }
        with torch.no_grad():
            return self.model.decode(batch_input)

    @torch.no_grad()
    def log_likelihood(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[float]:
        """
        Computing the likelihood of the encoder_input SMILES and decoder_input SMILES
        pairs.

        :param dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
        :param dataloader: If None -> dataloader
                will be retrieved from self.datamodule.
        :return: List with log-likelihoods of each reactant/product pairs.
        """

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        self.model.to(self.device)
        self.model.eval()

        log_likelihoods = []
        for batch in dataloader:
            batch = self.on_device(batch)
            output = self.model.forward(batch)
            log_probabilities = self.model.generator(output["model_output"])

            target_ids_lst = batch["decoder_input"].permute(1, 0)

            for target_ids, log_prob in zip(target_ids_lst[:, 1::], log_probabilities.permute(1, 0, 2)):
                llhs = 0.0
                for i_token, token in enumerate(target_ids):
                    llhs += log_prob[i_token, token].item()
                    break_condition = token == self.tokenizer["end"] or token == self.tokenizer["pad"]
                    if break_condition:
                        break

                log_likelihoods.append(llhs)
        return log_likelihoods

    def _prediction_setup(
        self,
        n_predictions: Optional[int] = None,
        sampler: Optional[SMILESSampler] = None,
        return_tokenized: bool = False,
    ) -> None:
        """
        Setup for Chemformer prediction.
        :param n_predictions: Number of sampled SMILES strings.
        :param sampler: An alternative sampler to use instead of the current self.sampler
        :param return_tokenized: Whether to return the tokenized beam search
                solutions instead of strings.
        """
        n_predictions: int = n_predictions if n_predictions else self.config.n_predictions
        sampler: SMILESSampler = sampler if sampler else self.sampler

        if sampler:
            self.model.sampler = sampler

        self.model.n_predictions = n_predictions
        self._return_tokenized = return_tokenized

    def _predict_batch(self, batch: StrDict) -> StrDict:
        """Return predictions for one batch"""
        predictions_batch, log_lhs_batch = self.model.sample_predictions(batch, return_tokenized=self._return_tokenized)
        if self.model.sampler:
            if self.model.sampler.sample_unique:
                predictions_batch = self.model.sampler.smiles_unique
                log_lhs_batch = self.model.sampler.log_lhs_unique

        predictions = {
            "predictions": predictions_batch,
            "log_likelihoods": log_lhs_batch,
            "ground_truth": batch.get("target_smiles"),
        }
        return predictions
