import os
from typing import List

import omegaconf as oc
from fastapi import FastAPI
from service_utils import get_classifier_predictions

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.utils.configs.chemformer.predict import Predict

app = FastAPI()

# Container for data, classes that can be loaded upon startup of the REST API
config = oc.OmegaConf.structured(Predict)
config.model_path = os.environ["CHEMFORMER_MODEL"]
config.model.type = "TransformerClassifier"
config.task = "classification"
config.model_hyperparams.batch_first = True
config.vocabulary_path = os.environ["CHEMFORMER_VOCAB"]
config.batch_size = 64
config.sampler = None
config.datamodule = None

global_items = {"model": Chemformer(config)}


@app.post("/classifier-api/predict")
def classifier_predict(smiles_list: List[str], n_predictions: int = 1):
    predictions, log_likelihoods = get_classifier_predictions(global_items["model"], smiles_list, n_predictions)

    output = []
    for (
        input_smiles,
        labels,
        log_lhs,
    ) in zip(smiles_list, predictions, log_likelihoods):
        output.append(
            {
                "input": input_smiles,
                "output": [int(val) for val in labels],
                "prob": [float(val) for val in log_lhs],
            }
        )
    return output


if __name__ == "__main__":
    import uvicorn

    port = os.environ.get("PORT") or 8231

    uvicorn.run(
        "classifier_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
