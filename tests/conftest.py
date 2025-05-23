import pandas as pd
import pytest

from aizynthmodels.chemformer.data import SynthesisDataModule
from aizynthmodels.chemformer.data.tokenizer import ChemformerTokenizer


@pytest.fixture
def vocabulary_path():
    return "tests/chemformer/data/simple_vocab.json"


@pytest.fixture
def setup_basic_tokenizer(vocabulary_path):
    return ChemformerTokenizer(filename=vocabulary_path)


@pytest.fixture
def seq2seq_data(tmpdir):
    filename = str(tmpdir / "seq2seq2_tmp.csv")
    data = pd.DataFrame(
        {
            "reactants": ["CC.O", "CCO.CC", "CC.O", "CCO.CC", "CC.O", "CNCO.CC"],
            "products": ["CCO", "CCOCC", "C(=C)O", "CC(=O)CC", "CCO", "CNCOCC"],
            "set": ["test", "test", "val", "train", "train", "train"],
        }
    )
    data.to_csv(filename, sep="\t", index=False)
    return filename


@pytest.fixture
def setup_synthesis_datamodule(seq2seq_data, setup_basic_tokenizer):
    datamodule = SynthesisDataModule(
        dataset_path=seq2seq_data, max_seq_len=128, tokenizer=setup_basic_tokenizer, batch_size=32
    )
    datamodule.setup()
    return datamodule
