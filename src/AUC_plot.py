from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

labels = ["RadNet-FtRadNet", "ImNet-FtImNet", "FtImNet-FtRadNet", "ImNet-RadNet"]

layer_list = ["conv1", "block1", "block2", "block3", "block4"]
"""
rc = {'figure.figsize':(7,5),
      'axes.facecolor':'white',
      'axes.grid' : False,
      'grid.color': '.9',
      }
plt.rcParams.update(rc)
"""

colors = ["#2b8cbe", "#a6bddb", "#e34a33", "#fdbb84", "#f0f0f0"]

df = pd.read_csv("results/AUC.csv")
df = df.sort_values(by=["target", "source"])
df["AUC"] = df["AUC"] * 100
f = sns.boxplot(
    data=df,
    x="target",
    y="AUC",
    hue=df[["source", "freeze"]].apply(tuple, axis=1),
    palette=colors,
    order=["thyroid", "breast", "chest", "mammograms", "knee", "isic", "pcam-small"],
)
sns.despine(bottom=False, left=False)
# plt.ylim([0.5, 1.0])
labels = [
    "ImageNet",
    "ImageNet freeze",
    "RadImageNet",
    "RadImageNet freeze",
    "Random initialization",
]
f.legend(
    title="Source dataset", frameon=False, loc="center left", bbox_to_anchor=(0.7, 0.2)
)
n = 0
for i in labels:
    f.legend_.texts[n].set_text(i)
    n += 1

plt.ylabel("AUC")
plt.xticks(rotation=45)
plt.xlabel("")
plt.savefig("results/AUC_figures/AUC.pdf", bbox_inches="tight")
plt.close()

df["mean"] = np.round(
    df.groupby(["target", "source", "freeze"])["AUC"].transform("mean"), 1
)
df["std"] = np.round(
    df.groupby(["target", "source", "freeze"])["AUC"].transform("std"), 2
)
df.to_csv("results/AUC_mean.csv")
