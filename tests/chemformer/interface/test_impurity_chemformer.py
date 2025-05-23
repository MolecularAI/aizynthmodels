import pytest
import pytorch_lightning as pl

from aizynthmodels.chemformer import ImpurityChemformer


@pytest.fixture
def impurity_config(default_config):
    config = default_config
    config.model_reagents = False
    config.n_predictions_baseline = 2
    config.n_predictions_non_baseline = 2
    config.n_predictions_purification = 1
    config.top_k_products = 1
    config.model_hyperparams.max_seq_len = 70
    pl.seed_everything(12)
    return config


@pytest.fixture
def impurity_data():
    return {
        "reactants_smiles": "CCO.CO",
        "solvent_smiles": "O.N",
    }


def test_model_initialization(impurity_config):
    model = ImpurityChemformer(impurity_config)
    assert isinstance(model, ImpurityChemformer)

    impurity_config.task = "backward_prediction"
    with pytest.raises(ValueError, match="Impurity prediction should be done with a forward prediction model."):
        model = ImpurityChemformer(impurity_config)


def test_predict_impurities(impurity_config, impurity_data):
    model = ImpurityChemformer(impurity_config)
    predictions = model.predict_impurities(**impurity_data)

    assert predictions.shape[0] == 12
    assert predictions.shape[1] == 6
    assert all(
        mode in predictions["mode"].values
        for mode in ["Reaction component", "Baseline", "Dimerization", "Over-reaction", "Solvent interaction"]
    )


def test_predict_impurities_with_purification(impurity_config, impurity_data):
    model = ImpurityChemformer(impurity_config)
    predictions = model.predict_impurities(**impurity_data, purification_solvent="standard")

    assert predictions.shape[0] == 16
    assert predictions.shape[1] == 6
    assert all(
        mode in predictions["mode"].values
        for mode in [
            "Reaction component",
            "Baseline",
            "Dimerization",
            "Over-reaction",
            "Solvent interaction",
            "Purification step reaction",
        ]
    )


def test_predict_impurities_with_reagents(impurity_config, impurity_data):
    model = ImpurityChemformer(impurity_config)

    impurity_data["reagent_smiles"] = "CCNCO"
    impurity_data["solvent_smiles"] = None
    predictions = model.predict_impurities(**impurity_data)

    assert predictions.shape[0] == 11
    assert predictions.shape[1] == 6
    assert all(
        mode in predictions["mode"].values
        for mode in ["Reaction component", "Baseline", "Reagent reaction", "Dimerization", "Over-reaction"]
    )


def test_predict_impurities_with_products(impurity_config, impurity_data):
    model = ImpurityChemformer(impurity_config)

    impurity_data["product_smiles"] = "CCNCCO"
    impurity_data["solvent_smiles"] = None
    predictions = model.predict_impurities(**impurity_data)

    assert predictions.shape[0] == 13
    assert predictions.shape[1] == 6
