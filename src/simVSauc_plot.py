from matplotlib import pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp

pd.options.mode.chained_assignment = None


def merge_df(file):
    Rad = pd.read_csv(file)
    Rad["target"] = Rad["model1"].str.split("-", expand=True)[0]
    Rad["target"] = Rad.target.replace({"pcam": "pcam-small"})
    AUC = pd.read_csv("results/AUC.csv")
    AUC_base = AUC[AUC["source"] == "random"]
    AUC_fine = AUC[AUC["source"] != "random"]
    AUC_fine["model1"] = AUC_fine[["target", "source", "freeze"]].apply(
        lambda x: "-".join(x.dropna().astype(str)), axis=1
    )
    Rad_merged = Rad.merge(AUC_base, on=["target", "fold"])
    Rad_merged = Rad_merged.merge(AUC_fine, on=["model1", "fold"])
    Rad_merged["AUC_diff"] = (Rad_merged["AUC_y"] - Rad_merged["AUC_x"]) / Rad_merged[
        "AUC_x"
    ]
    return Rad_merged


Im_df = merge_df("results/simVSaucImNet.csv")
Im_df = Im_df.sort_values(by=["target_y"])
Rad_df = merge_df("results/simVSaucRadNet.csv")
Rad_df = Rad_df.sort_values(by=["target_y"])

sns.set(font_scale=2.2)
sns.set_style(style="white")
fig, axes = plt.subplots(2, 5, figsize=(30, 12), sharey=True)

sns.scatterplot(
    ax=axes[0, 0], data=Im_df, x="conv1", y="AUC_diff", hue="target_y", s=100
)
axes[0, 0].legend([], [], frameon=False)
sns.regplot(
    ax=axes[0, 0],
    data=Im_df,
    x="conv1",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[0, 0].set_title("conv1")
axes[0, 0].set_ylabel("AUC difference")
axes[0, 0].set_xlabel("")

sns.scatterplot(
    ax=axes[0, 1], data=Im_df, x="block1", y="AUC_diff", hue="target_y", s=100
)
axes[0, 1].legend([], [], frameon=False)
g = sns.regplot(
    ax=axes[0, 1],
    data=Im_df,
    x="block1",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[0, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[0, 1].set_title("block1")
axes[0, 1].set_xlabel("")

sns.scatterplot(
    ax=axes[0, 2], data=Im_df, x="block2", y="AUC_diff", hue="target_y", s=100
)
axes[0, 2].legend([], [], frameon=False)
sns.regplot(
    ax=axes[0, 2],
    data=Im_df,
    x="block2",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[0, 2].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[0, 2].set_title("block2")
axes[0, 2].set_xlabel("")
sns.scatterplot(
    ax=axes[0, 3], data=Im_df, x="block3", y="AUC_diff", hue="target_y", s=100
)
axes[0, 3].legend([], [], frameon=False)
sns.regplot(
    ax=axes[0, 3],
    data=Im_df,
    x="block3",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[0, 3].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[0, 3].set_title("block3")
axes[0, 3].set_xlabel("")
axes[0, 3].set_xlim([0.38, 0.99])
sns.scatterplot(
    ax=axes[0, 4], data=Im_df, x="block4", y="AUC_diff", hue="target_y", s=100
)
axes[0, 4].legend([], [], frameon=False)
sns.regplot(
    ax=axes[0, 4],
    data=Im_df,
    x="block4",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[0, 4].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[0, 4].set_title("block4")
axes[0, 4].set_xlabel("")

sns.scatterplot(
    ax=axes[1, 0], data=Rad_df, x="conv1", y="AUC_diff", hue="target_y", s=100
)
axes[1, 0].legend([], [], frameon=False)
sns.regplot(
    ax=axes[1, 0],
    data=Rad_df,
    x="conv1",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[1, 0].set_ylabel("AUC difference")
axes[1, 0].set_xlabel("CCA")
sns.scatterplot(
    ax=axes[1, 1], data=Rad_df, x="block1", y="AUC_diff", hue="target_y", s=100
)
axes[1, 1].legend([], [], frameon=False)
sns.regplot(
    ax=axes[1, 1],
    data=Rad_df,
    x="block1",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[1, 1].set_xlabel("CCA")
sns.scatterplot(
    ax=axes[1, 2], data=Rad_df, x="block2", y="AUC_diff", hue="target_y", s=100
)
axes[1, 2].legend([], [], frameon=False)
sns.regplot(
    ax=axes[1, 2],
    data=Rad_df,
    x="block2",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[1, 2].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[1, 2].set_xlabel("CCA")
sns.scatterplot(
    ax=axes[1, 3], data=Rad_df, x="block3", y="AUC_diff", hue="target_y", s=100
)
axes[1, 3].legend([], [], frameon=False)
sns.regplot(
    ax=axes[1, 3],
    data=Rad_df,
    x="block3",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[1, 3].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[1, 3].set_xlabel("CCA")
sns.scatterplot(
    ax=axes[1, 4], data=Rad_df, x="block4", y="AUC_diff", hue="target_y", s=100
)
sns.regplot(
    ax=axes[1, 4],
    data=Rad_df,
    x="block4",
    y="AUC_diff",
    scatter=False,
    label="_nolegend_",
)
axes[1, 4].legend([], [], frameon=False)
axes[1, 4].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
axes[1, 4].set_xlabel("CCA")

handles, labels = axes[0, 4].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=7,
    loc="lower right",
    bbox_to_anchor=(0.82, -0.04),
    frameon=False,
    markerscale=2,
)

plt.subplots_adjust(wspace=0.05, hspace=0.25)
sns.despine(bottom=False, left=False)
plt.savefig("results/simVSauc_figures/combined.pdf", bbox_inches="tight")
plt.close()
