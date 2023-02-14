from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, help="target dataset")
args = parser.parse_args()

# CCA fine-tuning
model1 = [
    args.target + "-RadImageNet-freezeFalse",
    args.target + "-ImageNet-freezeFalse",
]
model2 = ["RadImageNet", "ImageNet"]

labels = [
    r"$\mathtt{RadImageNet}$-" + r"$\mathtt{RadImageNet}^\mathrm{FT}$",
    r"$\mathtt{ImageNet}$-" + r"$\mathtt{ImageNet}^\mathrm{FT}$",
]

layer_list = ["conv1", "block1", "block2", "block3", "block4"]

sns.set(font_scale=1.7)
palette = itertools.cycle(sns.color_palette())
sns.set_style(style="white")
df = pd.read_csv("results/CCA_" + args.target + ".csv")
for i, (m1, m2) in enumerate(zip(model1, model2)):
    filtered = df[(df["model1"] == m1) & (df["model2"] == m2)].T
    filtered.index.name = "layer"
    stacked = filtered[filtered.index.isin(layer_list)].stack()
    stacked = stacked.reset_index(name="CCA")
    stacked.drop("level_1", axis=1, inplace=True)
    c = next(palette)
    sns.lineplot(
        data=stacked, x="layer", y="CCA", errorbar="sd", label=labels[i], color=c
    )
    sns.pointplot(
        data=stacked,
        x="layer",
        y="CCA",
        errorbar="sd",
        label="_nolegend_",
        color=c,
        linestyles="",
    )

plt.ylim([0, 1])
# reordering the labels
handles, labels = plt.gca().get_legend_handles_labels()
# specify order
order = [1, 0]
plt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    frameon=False,
    bbox_to_anchor=(0.82, -0.1),
)
plt.ylabel("CCA similarity")
plt.xlabel("")
sns.despine(bottom=False, left=False)

plt.savefig("results/CCA_figures/CCA_" + args.target + "_ft.pdf", bbox_inches="tight")
plt.close()

# CCA ImageNet vs RadImageNet
model1 = [args.target + "-ImageNet-freezeFalse", "ImageNet", "random"]
model2 = [args.target + "-RadImageNet-freezeFalse", "RadImageNet", "random"]

sns.set(font_scale=1.7)
palette = itertools.cycle(sns.color_palette())
c = next(palette)
c = next(palette)
sns.set_style(style="white")
labels = [
    r"$\mathtt{ImageNet}^\mathrm{FT}$-" r"$\mathtt{RadImageNet}^\mathrm{FT}$",
    r"$\mathtt{ImageNet}$-" r"$\mathtt{RadImageNet}$",
    "Random-Random",
]
for i, (m1, m2) in enumerate(zip(model1, model2)):
    filtered = df[(df["model1"] == m1) & (df["model2"] == m2)].T
    filtered.index.name = "layer"
    stacked = filtered[filtered.index.isin(layer_list)].stack()
    stacked = stacked.reset_index(name="CCA")
    stacked.drop("level_1", axis=1, inplace=True)
    c = next(palette)
    sns.lineplot(
        data=stacked, x="layer", y="CCA", errorbar="sd", label=labels[i], color=c
    )
    sns.pointplot(
        data=stacked,
        x="layer",
        y="CCA",
        errorbar="sd",
        label="_nolegend_",
        color=c,
        linestyles="",
    )
plt.ylim([0, 1])
sns.despine(bottom=False, left=False)
plt.legend(frameon=False)
plt.yticks([])
plt.ylabel("")
plt.xlabel("")

plt.savefig(
    "results/CCA_figures/CCA_" + args.target + "_imagenet_radImagenet.pdf",
    bbox_inches="tight",
)
