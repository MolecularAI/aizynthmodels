import pytest
from omegaconf import OmegaConf

from aizynthmodels.chemformer.models.bart import BaseTransformer


def test_optimizer_not_in_config(default_config, setup_encoder):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    optimizer = model_hyperparams.pop("optimizer")
    optimizer["lr"] = optimizer.pop("learning_rate")
    optimizer["schedule"] = "cycle"
    model_hyperparams.update(optimizer)

    config = OmegaConf.create(model_hyperparams)
    transformer = BaseTransformer(config)

    assert transformer.optimizer == {
        "betas": [
            0.9,
            0.999,
        ],
        "weight_decay": 0.0,
        "learning_rate": 0.001,
        "scheduler": "cycle",
    }


def test_training_step(model_batch_setup):
    batch_input = model_batch_setup["batch_input"]
    batch_idx = model_batch_setup["batch_idx"]
    model = model_batch_setup["chemformer"].model

    output = model.training_step(batch_input, batch_idx)

    assert round(output.tolist(), 4) == 3.2243


def test_validation_step(model_batch_setup):
    batch_input = model_batch_setup["batch_input"]
    batch_idx = model_batch_setup["batch_idx"]
    model = model_batch_setup["chemformer"].model

    model.validation_step(batch_input, batch_idx)

    assert model.validation_step_outputs[0] == {
        "validation_loss": 3.224306583404541,
        "validation_token_accuracy": 0.07954545319080353,
        "accuracy_top_1": 0.0,
        "accuracy_top_3": 0.0,
    }

    model.on_validation_epoch_end()
    assert not model.validation_step_outputs


def test_test_step(model_batch_setup):
    batch_input = model_batch_setup["batch_input"]
    batch_idx = model_batch_setup["batch_idx"]
    model = model_batch_setup["chemformer"].model

    model.test_step(batch_input, batch_idx)

    assert round(model.test_step_outputs[0]["test_loss"], 4) == 3.2243


def test_configure_optimizers_cycle(default_config, setup_encoder):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    config = OmegaConf.create(model_hyperparams)
    transformer = BaseTransformer(config)
    optimizer, scheduler = transformer.configure_optimizers()

    assert round(optimizer[0].state_dict()["param_groups"][0]["lr"], 4) == 0.0008
    assert optimizer[0].state_dict()["param_groups"][0]["betas"] == (
        0.8688255099070633,
        0.999,
    )
    assert scheduler[0]["scheduler"].total_steps == model_hyperparams["num_steps"]


def test_configure_optimizers_transformer(default_config, setup_encoder):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model_hyperparams["optimizer"]["scheduler"] = "transformer"

    config = OmegaConf.create(model_hyperparams)
    transformer = BaseTransformer(config)
    optimizer, scheduler = transformer.configure_optimizers()

    assert round(optimizer[0].state_dict()["param_groups"][0]["lr"], 4) == 0.0002
    assert optimizer[0].state_dict()["param_groups"][0]["betas"] == (0.9, 0.999)
    assert len(scheduler[0]["scheduler"].get_lr()) == 1


def test_construct_lambda_lr_cycle(default_config, setup_encoder):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)

    config = OmegaConf.create(model_hyperparams)
    transformer = BaseTransformer(config)
    learning_rate = transformer.construct_lambda_lr(1)

    assert round(learning_rate, 4) == 0.0005


def test_construct_lambda_lr_transformer(default_config, setup_encoder):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model_hyperparams["optimizer"]["scheduler"] = "transformer"

    config = OmegaConf.create(model_hyperparams)
    transformer = BaseTransformer(config)
    learning_rate = transformer.construct_lambda_lr(1)

    assert round(learning_rate, 4) == 0.0002


def test_construct_lambda_lr_transformer_raise_error(default_config, setup_encoder):
    tokenizer, _, encoder = setup_encoder

    model_hyperparams = default_config.model_hyperparams
    model_hyperparams["vocabulary_size"] = len(tokenizer)
    model_hyperparams["optimizer"]["scheduler"] = "transformer"
    model_hyperparams["warm_up_steps"] = None

    config = OmegaConf.create(model_hyperparams)
    transformer = BaseTransformer(config)

    with pytest.raises(AssertionError, match="value for warm_up_steps is required"):
        transformer.construct_lambda_lr(3)
