import torch
from omegaconf import OmegaConf

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.chemformer.data import SynthesisDataModule


def test_set_datamodule(shared_datadir, default_config, setup_tokenizer):
    default_config.data_path = str(shared_datadir / "example_data_uspto.csv")
    default_config.datamodule = None

    chemformer = Chemformer(default_config)

    datamodule_config = OmegaConf.create({"type": "SynthesisDataModule"})
    chemformer.set_datamodule(datamodule_config=datamodule_config)
    assert chemformer.datamodule is not None

    chemformer.datamodule = None
    chemformer.set_datamodule()
    assert not chemformer.datamodule

    chemformer.datamodule = None
    synthesis_datamodule = SynthesisDataModule(
        dataset_path=str(shared_datadir / "example_data_uspto.csv"),
        tokenizer=setup_tokenizer(),
        batch_size=1,
        max_seq_len=100,
    )
    chemformer.set_datamodule(datamodule=synthesis_datamodule)
    assert chemformer.datamodule is not None


def test_log_likelihood(shared_datadir, default_config, setup_tokenizer):
    default_config.data_path = str(shared_datadir / "example_data_uspto.csv")
    default_config.datamodule = {"type": "SynthesisDataModule"}

    chemformer = Chemformer(default_config)
    log_likelihood = chemformer.log_likelihood()

    assert len(log_likelihood) == 3


def test_encode(shared_datadir, default_config):
    default_config.data_path = str(shared_datadir / "example_data_uspto.csv")
    default_config.datamodule = {"type": "SynthesisDataModule"}
    default_config.model_hyperparams["batch_first"] = False

    chemformer = Chemformer(default_config)
    encode_output = chemformer.encode()

    assert len(encode_output) == 3
    assert tuple(encode_output[0].shape) == (39, 4)


def test_encode_batch_first(shared_datadir, default_config):
    default_config.data_path = str(shared_datadir / "example_data_uspto.csv")
    default_config.datamodule = {"type": "SynthesisDataModule"}

    chemformer = Chemformer(default_config)
    encode_output = chemformer.encode()

    assert len(encode_output) == 39
    assert tuple(encode_output[0].shape) == (3, 4)


def test_decode(shared_datadir, default_config):
    default_config.data_path = str(shared_datadir / "example_data_uspto.csv")
    default_config.datamodule = {"type": "SynthesisDataModule"}

    chemformer = Chemformer(default_config)

    batch_input = {
        "memory": torch.Tensor(
            [
                [
                    [1.0, 0.15, 0.2, 0.5],
                ]
            ]
        ),
        "memory_pad_mask": torch.Tensor([[False]]),
        "decoder_input": torch.Tensor([[2]]).to(torch.int64),
    }

    decode_output = chemformer.decode(**batch_input)

    assert len(decode_output) == 1
    assert tuple(decode_output[0].shape) == (1, 4)
