import logging

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import ImpurityChemformer
from aizynthmodels.utils.configs.chemformer import impurity_prediction  # noqa: F401
from aizynthmodels.utils.hydra import custom_config


@hydra.main(
    version_base=None,
    config_name="predict_impurities",
)
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(1)

    logging.info("Running impurity prediction.")

    config.task = "forward_prediction"
    data = pd.read_csv(config.data_path, sep="\t").iloc[0]
    for col in data.index.values:
        if str(np.nan) == str(data[col]):
            data[col] = None

    chemformer = ImpurityChemformer(config)

    impurity_df = chemformer.predict_impurities(
        data.reactants,
        solvent_smiles=data.solvents,
        reagent_smiles=data.reagents,
        product_smiles=data.products,
        purification_solvent="standard",
    )

    impurity_df.to_csv(config.output_predictions, sep="\t", index=False)
    logging.info("Impurity prediction done.")


if __name__ == "__main__":
    main()
