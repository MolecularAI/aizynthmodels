# %%
from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd
from IPython.display import Markdown

pd.options.display.float_format = "{:,.2f}".format
print_ = lambda x: display(Markdown(x))  # noqa: E731, F821

# %% tags=["parameters"]
training_path = ""
metrics_path = ""

# %% [markdown]
"""
## Statistics on filter model training
"""

# %%
latest = sorted((Path(training_path) / "quick_filter").glob("version*"))[-1]
val_data = pd.read_csv(latest / "logged_train_metrics.csv", sep="\t")
sel = ~val_data["val_loss"].isna()
val_data = val_data[sel]
val_data.tail()

# %%
print_("Convergence of validation loss and accuracy")
fig = plt.figure()
ax = fig.gca()
ax2 = ax.twinx()
val_data.plot(x="epoch", y="val_loss", ax=ax, legend=False)
val_data.plot(x="epoch", y="val_binary_accuracy", style="g", ax=ax2, legend=False)
_ = fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

# %%
print_("Metrics at the last epoch")
for key, val in val_data.iloc[-1].to_dict().items():
    if key == "epoch":
        continue
    print_(f"- {key} = {val:.2f}")

# %% [markdown]
"""
## Statistics on test set
"""

# %%
test_metrics = pd.read_csv(metrics_path, sep="\t")
test_metrics.boxplot(rot=90)

# %%
test_metrics.describe()
