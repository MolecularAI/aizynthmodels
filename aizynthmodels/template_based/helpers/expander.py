"""
Command-line interface to AiZynthExpander

Only used by eval_one_step.py
"""

if __name__ == "__main__":
    import json
    import sys

    import pandas as pd
    from aizynthfinder.aizynthfinder import AiZynthExpander
    from rxnutils.chem.utils import split_rsmi
    from tqdm import tqdm

    ref_reactions_path, rsmi_column, top_n, config_path, output_path = sys.argv[1:]

    ref_reactions = pd.read_csv(ref_reactions_path, sep="\t")
    targets = [split_rsmi(rxn)[-1] for rxn in ref_reactions[rsmi_column]]  # config.columns.reaction_smiles]]

    expander = AiZynthExpander(configfile=config_path)
    expander.expansion_policy.select("default")
    expander_output = []
    for target in tqdm(targets):
        outcome = expander.do_expansion(target, int(top_n))
        outcome_list = [item.reaction_smiles().split(">>")[1] for item_list in outcome for item in item_list]
        expander_output.append(
            {
                "outcome": outcome_list,
                "non-applicable": expander.stats["non-applicable"],
            }
        )

    with open(output_path, "w") as fileobj:
        json.dump(expander_output, fileobj, indent=4)
