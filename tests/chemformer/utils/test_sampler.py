import pytest

from aizynthmodels.chemformer.sampler import BeamSearchSampler
from aizynthmodels.chemformer.utils.sampler import EOS, Criterion, LogicalAnd, LogicalOr, MaxLength


def test_criteria(model_batch_setup):
    chemformer = model_batch_setup["chemformer"]

    node_kwargs = {"tokenizer": chemformer.tokenizer, "device": "cpu", "batch_size": 16}

    node = BeamSearchSampler(**node_kwargs)
    node.initialize(chemformer.model, model_batch_setup["batch_input"], 1)

    criterion = Criterion()
    with pytest.raises(NotImplementedError, match="Not implemented"):
        criterion(node)

    max_len_crit = MaxLength(10)
    assert not max_len_crit(node)
    node.pos = 12
    assert max_len_crit(node)

    eos_crit = EOS()
    assert not eos_crit(node)

    and_crit = LogicalAnd([max_len_crit, eos_crit])
    or_crit = LogicalOr([max_len_crit, eos_crit])

    assert not and_crit(node)
    assert or_crit(node)
