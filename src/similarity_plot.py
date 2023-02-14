from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set(font_scale=1.2)
sns.set_style(style="white")
df = pd.read_csv("results/similarity.csv")
df = df[df["freeze"] == "freezeFalse"]
print(df)
f = sns.boxplot(
    data=df,
    x="target",
    y="value",
    hue=df[["freeze", "prob"]].apply(tuple, axis=1),
    order=["thyroid", "breast", "chest", "mammograms", "knee", "isic", "pcam-small"],
)
sns.despine(bottom=False, left=False)
labels = [
    "Observed predictions",
    "Independent predictions",
]  # , 'Prediction similarity (freeze)', 'Independent predictions (freeze)']
f.legend(title="", frameon=False, loc="best")
n = 0
for i in labels:
    f.legend_.texts[n].set_text(i)
    n += 1

plt.xticks(
    [0, 1, 2, 3, 4, 5, 6],
    labels=[
        r"$\mathtt{Thyroid}$",
        r"$\mathtt{Breast}$",
        r"$\mathtt{Chest}$",
        r"$\mathtt{Mammograms}$",
        r"$\mathtt{Knee}$",
        r"$\mathtt{ISIC}$",
        r"$\mathtt{PCam-small}$",
    ],
    rotation=45,
)
plt.ylabel("Prediction similarity")
plt.xlabel("")
plt.savefig("results/similarity_figures/similarity.pdf", bbox_inches="tight")
plt.close()
