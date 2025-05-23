"""Module containing routines to extract templates from reactions"""

import hydra
import pandas as pd
from omegaconf import DictConfig
from rxnutils.chem.reaction import ChemicalReaction, ReactionException
from rxnutils.data.batch_utils import read_csv_batch

# flake8: noqa: F401
from aizynthmodels.utils.configs.template_based import extract_templates
from aizynthmodels.utils.hydra import custom_config


def generate_templates(
    data: pd.DataFrame,
    radius: int,
    expand_ring: bool,
    expand_hetero: bool,
    ringbreaker_column: str,
    smiles_column: str,
) -> None:
    """
    Generate templates for the reaction in a given dataframe

    This function will add 3 columns to the dataframe
    * RetroTemplate: the extracted retro template
    * TemplateHash: a unique identifier based on fingerprint bits
    * TemplateError: if not None, will identicate a reason why the extraction failed

    :param data: the data with reactions
    :param radius: the radius to use, unless using Ringbreaker logic
    :param expand_ring: if True, will expand template with ring atoms
    :param expand_hetero: if True, will expand template with bonded heteroatoms
    :param ringbreaker_column: if given, will apply Rinbreaker logic to rows where this column is True
    :param smiles_column: the column with the atom-mapped reaction SMILES
    """

    def _row_apply(
        row: pd.Series,
        column: str,
        radius: int,
        expand_ring: bool,
        expand_hetero: bool,
        ringbreaker_column: str,
    ) -> pd.Series:
        rxn = ChemicalReaction(row[column], clean_smiles=False)
        general_error = {
            "RetroTemplate": None,
            "TemplateHash": None,
            "TemplateError": "General error",
        }

        if ringbreaker_column and row[ringbreaker_column]:
            expand_ring = True
            expand_hetero = True
            radius = 0
        elif ringbreaker_column and not row[ringbreaker_column]:
            expand_ring = False
            expand_hetero = False
        try:
            _, retro_template = rxn.generate_reaction_template(
                radius=radius, expand_ring=expand_ring, expand_hetero=expand_hetero
            )
        except ReactionException as err:
            general_error["TemplateError"] = str(err)
            return pd.Series(general_error)
        except Exception:
            general_error["TemplateError"] = "General error when generating template"
            return pd.Series(general_error)

        try:
            hash_ = retro_template.hash_from_bits()
        except Exception:
            general_error["TemplateError"] = "General error when generating template hash"
            return pd.Series(general_error)

        return pd.Series(
            {
                "RetroTemplate": retro_template.smarts,
                "TemplateHash": hash_,
                "TemplateError": None,
            }
        )

    template_data = data.apply(
        _row_apply,
        axis=1,
        radius=radius,
        expand_ring=expand_ring,
        expand_hetero=expand_hetero,
        ringbreaker_column=ringbreaker_column,
        column=smiles_column,
    )
    return data.assign(**{column: template_data[column] for column in template_data.columns})


@hydra.main(version_base=None, config_name="extract_templates")
@custom_config
def main(config: DictConfig) -> None:
    data = read_csv_batch(config.input_data, sep="\t", index_col=False, batch=config.batch)

    data = generate_templates(
        data,
        config.radius,
        config.expand_ring,
        config.expand_hetero,
        config.ringbreaker_column,
        config.smiles_column,
    )
    data.to_csv(config.output_data, index=False, sep="\t")


if __name__ == "__main__":
    main()
