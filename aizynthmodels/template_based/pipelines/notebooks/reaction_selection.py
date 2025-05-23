# %%
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from IPython.display import Markdown
from mendeleev import element as Element

pd.options.display.float_format = "{:,.2f}".format


def print_(x):
    display(Markdown(x))  # noqa: F821


# %% tags=["parameters"]
input_filename = ""
output_filename = ""

# %%
data = pd.read_csv(input_filename, sep="\t")


# %%
def make_stats_by_sources(data):
    sources = sorted(set(data.source))
    source_stats = defaultdict(list)
    for source in sources:
        source_stats["source"].append(source)
        df = data[data.source == source]
        source_stats["nreactions"].append(len(df))
        source_stats["% of total"].append(len(df) / len(data) * 100)
        unique_hashes = set(df["PseudoHash"])
        source_stats["n unique reactions"].append(len(unique_hashes))
        source_stats["% unique"].append(len(unique_hashes) / len(df) * 100)
        source_stats["% unclassified"].append(df["classification"].str.startswith("0.0 ").mean() * 100)
        df2 = df[~df.date.isna()]
        dates = pd.to_datetime(df2.date, infer_datetime_format=True)
        source_stats["from"].append(dates.min())
        source_stats["to"].append(dates.max())

    if len(sources) > 1:
        source_stats["source"].append("all")
        source_stats["nreactions"].append(sum(source_stats["nreactions"]))
        source_stats["% of total"].append("100")
        source_stats["n unique reactions"].append(len(set(data["PseudoHash"])))
        source_stats["% unique"].append(source_stats["n unique reactions"][-1] / source_stats["nreactions"][-1] * 100)
        source_stats["% unclassified"].append(data["classification"].str.startswith("0.0 ").mean() * 100)
        source_stats["from"].append(min(source_stats["from"]))
        source_stats["to"].append(max(source_stats["to"]))
    return source_stats


# %% [markdown]
"""
## Statistics on extracted and validated reactions
"""

# %%
sel = data["classification"].str.startswith("0.0 ")
print_(f"Total number of extracted reactions = {len(data)}")
print_(f"Number of unrecognized reactions = {sel.sum()} ({sel.mean()*100:.2f}%)")
nisotopes = (~data["Isotope"].isna()).sum()
print_(f"Number of reactions with isotope information = {nisotopes}")
ndeuterium = (data["Isotope"] == "2H").sum()
print_(f"Number of reactions with deuterium = {ndeuterium}")

bad_molecules = list()
sel = ~data.BadMolecules.isna()
for mol_list in data[sel].BadMolecules:
    bad_molecules.extend(mol_list.split(","))
sel2 = ~data.BadMolecules2.isna()
for mol_list in data[sel2].BadMolecules2:
    bad_molecules.extend(mol_list.split(","))
bad_molecules = set(bad_molecules)
print_(
    f"Identified {len(bad_molecules)} molecules that could not be sanitized "
    f"by RDKit this affected {sel.sum() + sel2.sum()} reactions"
)
print("")
pd.DataFrame(make_stats_by_sources(data))

# %%
ax = data.NReactants.value_counts().sort_index().plot.bar()
_ = ax.set_title("Distribution of number of reactants")

# %%
ax = data.NReagents.value_counts().sort_index().plot.bar()
_ = ax.set_title("Distribution of number of reagents")

# %%
ax = data.NProducts.value_counts().sort_index().plot.bar()
_ = ax.set_title("Distribution of number of products")

# %% [markdown]
"""
## Removing reactions based on these filters

- no. products should be 1
- no. reactants should be between 1 and 5
- no. un-mapped product atoms should be less than 3
- no. of widow atoms should be less than 5
- all reactants should be sanitizable
"""

# %%
NREACTANTS_LIMIT = 5
UNMAPPED_PROD_LIMIT = 3
WIDOWS_LIMIT = 5

sel_too_many_prod = data["NProducts"] > 1
sel_too_few_prod = (data["NProducts"] == 0) | (data["NMappedProducts"] == 0)
sel_too_few_react = (data["NReactants"] == 0) | (data["NMappedReactants"] == 0)
sel_too_many_reactants = data["NMappedReactants"] > NREACTANTS_LIMIT
sel_too_many_unmapped = data["UnmappedProdAtoms"] > UNMAPPED_PROD_LIMIT
sel_too_many_widows = data["WidowAtoms"] > WIDOWS_LIMIT
sel_unsani_react = data["HasUnsanitizableReactants"]

print_(f"Removing {sel_too_many_prod.sum()} reactions with more than one product")
print_(f"Removing {sel_too_few_prod.sum()} reactions without any product")
print_(f"Removing {sel_too_few_react.sum()} reactions without any reactant")
print_(f"Removing {sel_too_many_reactants.sum()} reactions with more than {NREACTANTS_LIMIT} reactants")
print_(f"Removing {sel_too_many_unmapped.sum()} reactions with more than {UNMAPPED_PROD_LIMIT} unmapped product atoms")
print_(f"Removing {sel_too_many_widows.sum()} reactions with more than {WIDOWS_LIMIT} widow atoms")
print_(f"Removing {sel_unsani_react.sum()} reactions with unsanitizable reactants")

data = data[
    (~sel_too_many_prod)
    & (~sel_too_few_prod)
    & (~sel_too_few_react)
    & (~sel_too_many_reactants)
    & (~sel_too_many_unmapped)
    & (~sel_too_many_widows)
    & (~sel_unsani_react)
]

# %% [markdown]
"""
## Removing reactions based on these filters

- the reactants must be different than the products
- no wild card atoms
- the no the product atoms should be between the 3% and 97% percentile
- there should not be any un-mapped radicals
- the elements in the reactants should be likely
- the reaction class should not be 11. or 12.
- the CGR should be possible to create
- the number of dynamic bonds should not be more than 10
- the reaction should not have more than 2 sibling reactions
- the reaction should not have more than 12 rings
"""


# %%
def unchanged_reactions(row):
    reactants, products = row["PseudoHash"].split(">>")
    return reactants == products


sel_unchanged = data.apply(unchanged_reactions, axis=1)
print_(f"Removing {sel_unchanged.sum()} reactions with the same reactants and products")

sel_wildcard_atom = data["rsmi_processed"].str.contains("*", regex=False)
print_(f"Removing {sel_wildcard_atom.sum()} reactions with wild card atoms")

percentile03 = data.ProductSize.quantile(0.03)
percentile97 = data.ProductSize.quantile(0.97)
sel_small = data.ProductSize < percentile03
sel_big = data.ProductSize > percentile97
print_(f"Removing {sel_small.sum()} reactions with product size smaller than {percentile03} (3% percentile)")
print_(f"Removing {sel_big.sum()} reactions with product size larger than {percentile97} (97% percentile)")

sel_radical = data["HasUnmappedRadicalAtom"]
print_(f"Removing {sel_radical.sum()} reactions with unmapped radical")

# %%
LIKELIHOOD_LIMIT = 1e-5


def reject_based_on_likelihood(row, likelihoods, limit):
    if row.classification != "0.0 Unrecognized":
        return False
    atomic_counts = json.loads(row.ElementCount)
    return any(likelihoods[atomic_number] < limit for atomic_number in atomic_counts.keys())


counts_by_atomic_number = defaultdict(int)
for count_str in data.ElementCount.values:
    for atomic_number, count in json.loads(count_str).items():
        counts_by_atomic_number[atomic_number] += count

sum_counts = sum(counts_by_atomic_number.values())
likelihoods = {key: value / sum_counts for key, value in counts_by_atomic_number.items()}
sel_likelihood = data.apply(reject_based_on_likelihood, axis=1, likelihoods=likelihoods, limit=LIKELIHOOD_LIMIT)
print_(f"Removing {sel_likelihood.sum()} reactions unusual elements in mapped reactants")
elements = ", ".join(
    sorted({Element(int(key)).symbol for key, value in likelihoods.items() if key != "0" and value < LIKELIHOOD_LIMIT})
)
print_(f"Elements considered to be unlikely: {elements}")

# %%
unwanted_classes = data.classification.str.startswith(("11.", "12."))
print_(f"Removing {unwanted_classes.sum()} reactions with unwanted reaction classes (11., 12.)")

sel_cgr_creation = ~data.CGRCreated
print_(f"Removing {sel_cgr_creation.sum()} reactions where the CGR could not be created")

DYNAMIC_BOND_LIMIT = 10
sel_dynamic_bond = data.NDynamicBonds > DYNAMIC_BOND_LIMIT
print_(
    f"Removing {sel_dynamic_bond.sum()} reactions where the number of dynamic "
    f"bonds are greater than {DYNAMIC_BOND_LIMIT}"
)

# %%
info = data["id"].str.extract(r"_P(?P<product_no>\d)$", expand=False)
prod_val = info[~info.isna()].astype(int)
sib_sel = np.zeros(len(data)).astype("bool")
if len(prod_val) > 0 and "document_id" in data.columns:
    prod_sel = prod_val > 2
    data_sel = data.index.isin(prod_val[prod_sel].index)
    unique_doc = data[data_sel].apply(lambda row: row["id"].split(f"_{row['source']}_")[0], axis=1).unique()
    sib_sel = data["document_id"].isin(unique_doc)
    print_(f"Removing {sib_sel.sum()} reactions with with more than 2 siblings")

# %%
NRING_LIMIT = 12
nrings_sel = data["MaxRings"] > NRING_LIMIT
print_(f"Removing {nrings_sel.sum()} reactions with with more than {NRING_LIMIT} rings")


# %%
data = data[
    (~sel_likelihood)
    & (~sel_small)
    & (~sel_big)
    & (~sel_wildcard_atom)
    & (~sel_unchanged)
    & (~sel_radical)
    & (~unwanted_classes)
    & (~sel_cgr_creation)
    & (~sel_dynamic_bond)
    & (~sib_sel)
    & (~nrings_sel)
]

# %%
print_("Statistics by source after filtering")
pd.DataFrame(make_stats_by_sources(data))

# %% [markdown]
"""
## Ring breaker
Finding reactions to be treated with ring breaker
"""

# %%
data["RingBreaker"] = (
    data["RingBondMade"] & (data["NRingChange"] >= 1) & (data["RingMadeSize"] > 2) & (data["RingMadeSize"] < 8)
)
nringbreaker = data["RingBreaker"].sum()
print_(f"Marked {nringbreaker} ({data['RingBreaker'].mean()*100:.2f}) reactions for ring breaker")
nring_sizes = [(data[data["RingBreaker"]]["RingMadeSize"] == n).sum() for n in range(3, 8)]
pd.DataFrame(
    {
        "size of ring formed": range(3, 8),
        "n reactions": nring_sizes,
        "% of total": [n / nringbreaker * 100 for n in nring_sizes],
    }
)

# %% [markdown]
"""
## Finalizing
"""

# %%
filename = output_filename.replace(".csv", "_all.csv")
print_(f"Saving all selected reactions with all columns to {filename}")
data.to_csv(filename, index=False, sep="\t")

# %%
hash_to_id = data.groupby("PseudoHash")["id"].apply(lambda x: list(x.values)).to_dict()
filename = output_filename.replace(".csv", "_ids.json")
with open(filename, "w") as fileobj:
    json.dump(hash_to_id, fileobj)
print_(f"Saving translation from reaction hash to reaction connect id to {filename}")

# %%
print_("Replacing classification with most common classification for each reaction")
most_common_classification_by_hash = (
    data.groupby("PseudoHash")["classification"].apply(lambda x: Counter(x.values).most_common(1)[0][0]).to_dict()
)
data["classification"] = data.apply(lambda row: most_common_classification_by_hash[row["PseudoHash"]], axis=1)


# %%
data.rename(columns={"rsmi_processed": "RxnSmilesClean"}, inplace=True)
data = data[["RxnSmilesClean", "PseudoHash", "classification", "RingBreaker"]]
data = data.drop_duplicates(subset="PseudoHash")
print_(
    f"Saving clean reaction SMILES, reaction hash, classification and "
    f"ring breaker flag for selected reactions to {output_filename}"
)
data.to_csv(output_filename, index=False, sep="\t")
