# %%
from collections import Counter, defaultdict

import matplotlib.pylab as plt
import pandas as pd
from omegaconf import OmegaConf
from aizynthmodels.utils.hydra import load_config
from aizynthmodels.utils.pipelines.files import prefix_filename
from aizynthmodels.utils.configs.template_based.pipelines import TemplatePipeline
from IPython.display import Markdown

pd.options.display.float_format = "{:,.2f}".format


def print_(x):
    display(Markdown(x))  # noqa: F821


# %% tags=["parameters"]
input_filename = ""
output_prefix = ""
output_postfix = ""
min_occurrence = 3
config_filename = ""

# %%
data = pd.read_csv(input_filename, sep="\t")
config = load_config(config_filename, TemplatePipeline)
col_config = config.template_library_columns

# %% [markdown]
"""
## Statistics on extracted and validated templates
"""

# %%
sel = data["TemplateError"].isna()
print_(f"Total number of extracted templates = {sel.sum()} ({sel.mean():.2f}%)")
print_(f"Number of unique templates = {len(set(data[sel]['TemplateHash']))}")
print_(f"Number of templates failures = {len(data)-sel.sum()} ({1-sel.sum()/len(data):.5f}%)")

counts = Counter(data[data["RetroTemplate"].isna()]["TemplateError"])
cat_counts = defaultdict(int)
for key, count in counts.items():
    if "too many unmapped atoms" in key:
        cat_counts["Too many unmapped atoms"] += count
    elif "consistent tetrahedral mapping" in key:
        cat_counts["Consistent tetrahedral mapping"] += count
    else:
        key = (
            key.replace("Template generation failed: ", "")
            .replace("Template generation failed with message:", "")
            .replace(
                "Template generation failed with message: Template generation failed:",
                "",
            )
            .replace("\n", "\t")
        )
        cat_counts[key.strip()] += count
print_("Errors and counts: ")
for key, count in cat_counts.items():
    print_(f"- {key}: {count}")

print("")

# %%
counts = data[data["TemplateError"].isna()].groupby("TemplateHash", sort=False).size().value_counts().to_dict()
ranges = [
    (0, 5),
    (5, 10),
    (10, 20),
    (20, 50),
    (50, 100),
    (100, 500),
    (500, max(counts.keys())),
]
count_ranges = []
for start, end in ranges:
    count_ranges.append(sum(counts.get(idx, 0) for idx in range(start, end)))
labels = [f"{start}-{end}" for start, end in ranges[:-1]] + [f"{ranges[-1][0]}-"]
plt.gca().bar(height=count_ranges, x=range(1, len(ranges) + 1))
plt.gca().set_title("Distribution of template occurence")
plt.gca().set_xticks(range(1, len(ranges) + 1))
_ = plt.gca().set_xticklabels(labels)

# %%
ax = data[data["TemplateError"].isna()].TemplateGivesNOutcomes.value_counts().sort_index().plot.bar()
_ = ax.set_title("Distribution of number of outcomes")

# %%
ncorrect = data[data["TemplateError"].isna()].TemplateGivesTrueReactants
print_(f"Number of templates that reproduces the correct reactants: {ncorrect.sum()} ({ncorrect.mean()*100:.2f}%)")
nother = data[data["TemplateError"].isna()].TemplateGivesOtherReactants
print_(f"Number of templates that produces other reactants: {nother.sum()} ({nother.mean()*100:.2f}%)")

# %% [markdown]
"""
## Removing templates based on filters
"""

# %%
sel_too_many_prod = data["TemplateError"].isna() & (data.nproducts > 1)
sel_failures = data["RetroTemplate"].isna()
sel_no_reprod = data["TemplateError"].isna() & (~data.TemplateGivesTrueReactants)
data = data[(~sel_too_many_prod) & (~sel_failures) & (~sel_no_reprod)]
print_(f"Removing {sel_too_many_prod.sum()} reactions with more than one product in template")
print_(f"Removing {sel_failures.sum()} reactions with failed template generation")
print_(f"Removing {sel_no_reprod.sum()} reactions with template unable to reproduce correct reactants")


# %%
print_(f"Number of unique templates before occurence filtering = {len(set(data['TemplateHash']))}")
print_(f"Removing all templates with occurrence less than {min_occurrence}")
template_group = data.groupby("TemplateHash")
template_group = template_group.size().sort_values(ascending=False)
min_index = template_group[template_group >= int(min_occurrence)].index
data = data[data["TemplateHash"].isin(min_index)]

# %% [markdown]
"""
## Finalizing
"""

# %%
data = data[
    [
        "RxnSmilesClean",
        "PseudoHash",
        "classification",
        "RingBreaker",
        "RetroTemplate",
        "TemplateHash",
    ]
]
data.rename(
    columns={
        "RxnSmilesClean": col_config.reaction_smiles,
        "PseudoHash": col_config.reaction_hash,
        "classification": col_config.classification,
        "RingBreaker": col_config.ring_breaker,
        "RetroTemplate": col_config.retro_template,
        "TemplateHash": col_config.template_hash,
    },
    inplace=True,
)
# %%
filename = prefix_filename(output_prefix, output_postfix)
print_(f"Total number of selected templates = {len(data)}")
print_(f"Number of unique templates = {len(set(data[col_config.template_hash]))}")
print_(f"Saving selected templates to {filename}")
data.to_csv(filename, index=False, sep="\t")

# %%
data = data[data[col_config.ring_breaker]]
filename = prefix_filename(output_prefix, "ringbreaker_" + output_postfix)
print_(f"Total number of selected ring breaker templates = {len(data)}")
print_(f"Number of unique ring breaker templates = {len(set(data[col_config.template_hash]))}")
print_(f"Saving selected ring breaker templates to {filename}")
data.to_csv(filename, index=False, sep="\t")
