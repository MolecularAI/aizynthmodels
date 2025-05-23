from omegaconf import OmegaConf

from aizynthmodels.chemformer.utils.data import construct_datamodule_kwargs


def test_construct_datamodule_kwargs(setup_tokenizer):
    dummy_config = OmegaConf.create({"task": "test", "data_path": "", "batch_size": 0})
    datamodule_kwargs = construct_datamodule_kwargs(dummy_config, setup_tokenizer())

    assert "tokenizer" in datamodule_kwargs
    assert isinstance(datamodule_kwargs["tokenizer"], type(setup_tokenizer()))

    datamodule_kwargs.pop("tokenizer")
    assert datamodule_kwargs == {
        "reverse": False,
        "max_seq_len": 512,
        "masker": None,
        "augment_prob": None,
        "dataset_path": "",
        "batch_size": 0,
        "add_sep_token": False,
        "i_chunk": 0,
        "n_chunks": 1,
    }
