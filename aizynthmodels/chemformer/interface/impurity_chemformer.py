import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rxnutils.chem.utils import split_smiles_from_reaction

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.chemformer.utils.impurity_chemformer import (
    default_purification_agents,
    get_dimerization_data,
    get_overreaction_data,
    get_purification_data,
    get_reaction_components_data,
    get_solvent_interaction_data,
    setup_datamodule,
    unravel_impurity_predictions,
)


class ImpurityChemformer(Chemformer):
    """
    Class for building an impurity prediction Chemformer model based on the Chemformer
    forward synthesis model. This class is only used for inference.
    The provided forward model should have been fine-tuned using the Chemformer class.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        """
        :param config: hydra config file with Chemformer arguments.

            Additional arguments for ImpurityChemformer:
                model_reagents (bool): Whether to use the model trained with reagents.
        """

        if config.task != "forward_prediction":
            raise ValueError("Impurity prediction should be done with a forward prediction model.")

        super().__init__(config)

        self.model_reagents = config.model_reagents
        self.n_predictions_baseline = config.n_predictions_baseline
        self.n_predictions_non_baseline = config.n_predictions_non_baseline
        self.n_predictions_purification = config.n_predictions_purification

        self.top_k_products = config.top_k_products
        self._logger = logging.getLogger("impurity-predictor")

    def predict_impurities(
        self,
        reactants_smiles: str,
        solvent_smiles: Optional[str] = None,
        reagent_smiles: Optional[str] = None,
        product_smiles: Optional[str] = None,
        purification_solvent: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict all possible product SMILES given reactants, reagents and solvent.

        :param reactants_smiles: input reactants
        :param solvent_smiles: solvent used in experiment
        :param reagent_smiles: reagents used in experiment
        :param product_smiles: the main product
        :param purification_solvent: the experiment mode to use for determining
            purification solvents. Either "standard" or "supercritical_fluid_ms". If
            set to None, the purification step will be skipped.
        :return: A dataframe with impurity predictions, corresponding reactants,
            log-likelihoods and prediction-mode origins.
        """

        impurity_df = self._reaction_step_impurities(reactants_smiles, reagent_smiles, solvent_smiles, product_smiles)

        if not purification_solvent:
            return unravel_impurity_predictions(impurity_df)

        self.purification_agents = default_purification_agents(purification_solvent)

        impurity_df = self._purification_step_impurities(impurity_df)
        return impurity_df

    def _baseline_predictions(
        self, reactants: List[str], reagents: List[str], product_smiles: Optional[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Predicting the baseline products, and the impurities from reactants reacting with
        reagents.

        :param reactants: list of reactants
        :param reagents: reagents used in experiment
        :param product_smiles: the main (expected) product
        :return: A tuple with the
            1. dataframe with baseline predictions, corresponding reactants,
                log-likelihoods and prediction-mode origins.
            2. the reagents to use in the impurity predictions.
            3. the products to use in the impurity predictions.
        """
        self._logger.info("Baseline predictions.")

        if self.model_reagents or not reagents:
            reacting_smiles = [".".join(reactants + reagents)]
            baseline_mode = ["Baseline"]
        else:
            reacting_smiles = [".".join(reactants), ".".join(reactants + reagents)]
            baseline_mode = ["Baseline", "Reagent reaction"]
            reagents = []  # Will not consider reagents in any other predictions

        # Set model beam size for baseline predictions
        self.config.n_predictions = self.n_predictions_baseline
        baseline_predictions, baseline_log_lhs = self._single_prediction(reacting_smiles)

        # Set model beam size for non-baseline predictions
        self.config.n_predictions = self.n_predictions_non_baseline

        if product_smiles:
            products = split_smiles_from_reaction(product_smiles)
        else:
            products = baseline_predictions[0][0 : self.top_k_products]  # noqa: E203

        baseline_df = pd.DataFrame(
            {
                "reactants": reacting_smiles[0:1],
                "mode": baseline_mode[0],
                "predicted_impurity": list(baseline_predictions[0:1]),
                "log_likelihood": list(baseline_log_lhs[0:1]),
                "target_smiles": [product_smiles],
            }
        )

        if "Reagent reaction" in baseline_mode:
            reagent_df = pd.DataFrame(
                {
                    "reactants": reacting_smiles[1::],
                    "mode": baseline_mode[1],
                    "predicted_impurity": list(baseline_predictions[1::]),
                    "log_likelihood": list(baseline_log_lhs[1::]),
                }
            )
            baseline_df = pd.concat([baseline_df, reagent_df], axis=0, ignore_index=True)

        return baseline_df, reagents, products

    def _first_step_impurities(self, reactants: List[str], products: List[str], reagents: List[str]) -> pd.DataFrame:
        """
        Predicting the first step impurities, corresponding to dimerization,
        over-reaction and subset of reactants.

        :param reactants: list of reactants
        :param products: the (expected or predicted) list of products
        :param reagents: reagents used in experiment (empty if self.model_reagents=False)
        :return: A dataframe with impurity predictions, corresponding reactants,
            log-likelihoods and prediction-mode origins.
        """
        # Collect possible impurity reactants
        self._logger.info("Collecting possible impurity reactants.")

        # Mode 2: Dimerization
        dimers_df = get_dimerization_data(reactants, products, reagents, self.model_reagents)

        # Mode 3: Over-reaction predictions + subset of reactants available
        overreaction_df = get_overreaction_data(reactants, products, reagents, self.model_reagents)

        impurity_df = pd.concat([dimers_df, overreaction_df], axis=0, ignore_index=True)

        impurity_df.drop_duplicates(subset=["reactants"], keep="first", ignore_index=True, inplace=True)

        self._logger.info("Predicting first-step impurities.")
        (
            impurity_predictions,
            impurity_log_lhs,
        ) = self._single_prediction(impurity_df["reactants"].values)

        impurity_df = impurity_df.assign(
            **{
                "predicted_impurity": impurity_predictions,
                "log_likelihood": impurity_log_lhs,
            }
        )
        return impurity_df

    def _predicted_impurities(self, impurity_df: pd.DataFrame) -> List[str]:
        """
        Returns a list with the predicted impurities (excluding reaction components).
        """
        impurity_df_flat = unravel_impurity_predictions(impurity_df)
        impurity_df_flat = impurity_df_flat.iloc[impurity_df_flat["mode"].values != "Reaction component"]
        return list(impurity_df_flat.predicted_impurity.values)

    def _purification_step_impurities(self, impurity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict possible impurities during the purification step in mobile phase.

        :param impurity_df: predicted impurities from the reaction step
        :return: A concatenated dataframe with impurity predictions from both the
            reaction step and the purification step, with corresponding log-likelihoods
            and prediction-mode origins.
        """
        predicted_impurities = self._predicted_impurities(impurity_df)

        purification_df = get_purification_data(predicted_impurities, self.purification_agents)
        purification_df.drop_duplicates(subset=["reactants"], keep="first", ignore_index=True, inplace=True)

        self._logger.info("Predicting reactions in purification step.")
        self.config.n_predictions = self.n_predictions_purification
        (
            purification_predictions,
            purification_log_lhs,
        ) = self._single_prediction(purification_df["reactants"].values)

        purification_df = purification_df.assign(
            **{
                "predicted_impurity": purification_predictions,
                "log_likelihood": purification_log_lhs,
            }
        )

        impurity_df = pd.concat(
            [impurity_df, purification_df],
            axis=0,
            ignore_index=True,
        )

        impurity_df = unravel_impurity_predictions(impurity_df)

        self._logger.info("Purification impurity prediction done.")
        return impurity_df

    def _reaction_step_impurities(
        self,
        reactants_smiles: str,
        reagent_smiles: Optional[str],
        solvent_smiles: Optional[str],
        product_smiles: Optional[str],
    ) -> pd.DataFrame:
        """
        Predict impurities using different possible modes during the actual reaction,
        given reactants, reagents and solvent.

        :param reactants_smiles: input reactants
        :param solvent_smiles: solvent used in experiment
        :param reagent_smiles: reagents used in experiment
        :param product_smiles: the main product
        :return: A dataframe with (raw) impurity predictions, corresponding log-likelihoods and
            prediction-mode origins.
        """
        reactants = split_smiles_from_reaction(reactants_smiles)

        if not solvent_smiles:
            solvent = []
        else:
            solvent = split_smiles_from_reaction(solvent_smiles)

        # Reagents are always considered as a whole set of molecules/smiles
        if not reagent_smiles:
            reagents = []
        else:
            reagents = [reagent_smiles]

        if reagents:
            reaction_components_df = get_reaction_components_data(
                reactants + split_smiles_from_reaction(reagents[0]) + solvent
            )
        else:
            reaction_components_df = get_reaction_components_data(reactants + solvent)

        # Baseline predictions (Mode 1)
        baseline_df, reagents, products = self._baseline_predictions(reactants, reagents, product_smiles)

        # First step impurities (Modes 2-3)
        impurity_df = self._first_step_impurities(reactants, products, reagents)

        impurity_df = pd.concat(
            [baseline_df, impurity_df],
            axis=0,
            ignore_index=True,
        )

        impurity_df.drop_duplicates(subset=["reactants"], keep="first", ignore_index=True, inplace=True)
        impurity_df = pd.concat([reaction_components_df, impurity_df], axis=0, ignore_index=True)

        # Solvent interaction impurities (mode 4)
        impurity_df = self._solvent_interaction_impurities(reactants, impurity_df, solvent, reagents)

        self._logger.info("Reaction step impurity prediction done.")
        return impurity_df

    def _single_prediction(self, smiles_list: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Given a list of input reactant SMILES, create a dataloader and sample
        product SMILES.

        :param smiles_list: list of input SMILES to the prediction
        :return: Tuple with the unique (sampled SMILES, log-likelihoods)
        """
        datamodule = setup_datamodule(smiles_list, self.tokenizer, self.config.batch_size)
        dataloader = self.get_dataloader("full", datamodule)

        output = self.predict(dataloader=dataloader)
        return output["predictions"], output["log_likelihoods"]

    def _solvent_interaction_impurities(
        self,
        reactants: List[str],
        impurity_df: pd.DataFrame,
        solvent: List[str],
        reagents: List[str],
    ) -> pd.DataFrame:
        """
        Predict impurities from solvent interaction. Solvent can interact with any of the
        predicted impurities in the reaction step (reactants, reagents, products,
        and impurities).

        :param impurity_df: contains the predicted impurities
        :param solvent: solvent used in experiment
        :param reagents: reagents used in experiment
        :return A dataframe with (raw) impurity predictions, corresponding log-likelihoods and
            prediction-mode origins.
        """
        if not solvent:
            return impurity_df

        predicted_impurities = self._predicted_impurities(impurity_df)
        solvent_interaction_df = get_solvent_interaction_data(
            reactants, predicted_impurities, solvent, reagents, self.model_reagents
        )
        solvent_interaction_df.drop_duplicates(subset=["reactants"], keep="first", ignore_index=True, inplace=True)

        self._logger.info("Predicting solvent interaction impurities.")
        (
            solvent_interaction_predictions,
            solvent_interaction_log_lhs,
        ) = self._single_prediction(solvent_interaction_df["reactants"].values)

        solvent_interaction_df = solvent_interaction_df.assign(
            **{
                "predicted_impurity": solvent_interaction_predictions,
                "log_likelihood": solvent_interaction_log_lhs,
            }
        )

        impurity_df = pd.concat([impurity_df, solvent_interaction_df], axis=0, ignore_index=True)
        return impurity_df
