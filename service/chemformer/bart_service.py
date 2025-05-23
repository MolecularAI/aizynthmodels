import os
from typing import List

import omegaconf as oc
from fastapi import FastAPI
from service_utils import calculate_llhs, estimate_compound_llhs, get_predictions

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.utils.configs.chemformer.predict import Predict

app = FastAPI()

# Container for data, classes that can be loaded upon startup of the REST API
config = oc.OmegaConf.structured(Predict)

config.batch_size = 64
config.model_path = os.environ["CHEMFORMER_MODEL"]
config.task = os.environ["CHEMFORMER_TASK"]
config.vocabulary_path = os.environ["CHEMFORMER_VOCAB"]
config.datamodule = None

global_items = {"chemformer": Chemformer(config)}


@app.post("/chemformer-api/predict")
def predict(smiles_list: List[str], n_predictions: int = 10):
    smiles, log_lhs, original_smiles = get_predictions(global_items["chemformer"], smiles_list, n_predictions)

    output = []
    for item_pred, item_lhs, item_smiles in zip(smiles, log_lhs, original_smiles):
        output.append(
            {
                "input": item_smiles,
                "output": list(item_pred),
                "lhs": [float(val) for val in item_lhs],
            }
        )
    return output


@app.post("/chemformer-api/log_likelihood")
def log_likelihood(reactants: List[str], products: List[str]):
    log_lhs = calculate_llhs(global_items["chemformer"], reactants, products)

    output = []
    for prod_smi, react_smi, llhs in zip(products, reactants, log_lhs):
        output.append(
            {
                "product_smiles": str(prod_smi),
                "reactant_smiles": str(react_smi),
                "log_likelihood": float(llhs),
            }
        )
    return output


@app.post("/chemformer-api/compound_log_likelihood")
def compound_log_likelihood(reactants: List[str], products: List[str], n_augments: int = 10):
    log_lhs = estimate_compound_llhs(global_items["chemformer"], reactants, products, n_augments=n_augments)

    output = []
    for prod_smi, react_smi, llhs in zip(products, reactants, log_lhs):
        output.append(
            {
                "product_smiles": str(prod_smi),
                "reactant_smiles": str(react_smi),
                "log_likelihood": float(llhs),
            }
        )
    return output


if __name__ == "__main__":
    import uvicorn

    port = os.environ.get("PORT") or 8003

    uvicorn.run(
        "bart_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
