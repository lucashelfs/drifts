import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from usp_stream_datasets import insects_datasets, load_insect_dataset
from utils import generate_filenames, open_metadata_file

sns.set(rc={"figure.figsize": (12, 6)})
tested_datasets = []

p_value = 0.05


def fetch_change_points(dataset_name: str) -> list:
    """Fetch change point list for a requested Dataset."""
    return insects_datasets[dataset_name].get("change_point", [])


def plot_p_values(series, p_value):
    g = sns.lineplot(data=series)
    g.axhline(p_value, ls="--", c="red")
    plt.figure()
    plt.show()


def plot_dataset_change_points(series, points, p_value):
    g = sns.lineplot(data=series)
    g.axhline(p_value, ls="--", c="red")
    for point in points:
        g.axvline(point, ls="--", c="yellow")
    plt.figure()
    plt.show()


def plot_multiple_p_values(dataset_name, p_value, save=False):
    csv_file, _, plot_file = generate_filenames(dataset_name)
    df_analysis = pd.read_csv(csv_file)
    df_analysis["action"] = np.where(
        df_analysis["p_value"] <= p_value, "reject", "accept"
    )
    attr_list = list(
        df_analysis[df_analysis["action"] == "accept"]
        .attr.value_counts()
        .to_dict()
        .keys()
    )[:5]
    df_plot = df_analysis[(df_analysis["attr"].isin(attr_list))][
        ["p_value", "attr", "start"]
    ]
    change_points = fetch_change_points(dataset_name)
    plt.clf()
    title = f"p-values for {dataset_name} dataset"
    plt.title(title)
    g = sns.lineplot(x="start", y="p_value", hue="attr", data=df_plot)
    g.axhline(p_value, ls="--", c="red")
    if change_points:
        for change_point in change_points:
            g.axvline(
                change_point, ls="--", c="yellow"
            )  # we need to slide this to the front. is it possible to do it?
    if save:
        plt.savefig(plot_file)
    plt.show()


for dataset in insects_datasets.keys():
    csv_file, metadata_file, _ = generate_filenames(dataset)
    if os.path.isfile(csv_file):
        tested_datasets.append(dataset)


dataset_infos = []


for dataset in tested_datasets:
    csv_file, metadata_file, _ = generate_filenames(dataset)
    df = pd.read_csv(csv_file)
    dataset_infos.append({"dataset": dataset, "rows": df.shape[0]})

df_rows = pd.DataFrame(dataset_infos)
df_rows = df_rows.sort_values("rows")


# # Plot and Save
# for dataset in df_rows.dataset.unique()[:5]:
#     plot_multiple_p_values(dataset, p_value, save=False)


# def plot_original_df(dataset_name, attr="Att1", species=None):
#     """Load dataframe from the insects datasets."""
#     df = load_insect_dataset(insects_datasets[dataset_name]["filename"])
#     if species:
#         df = df[df["class"] == species].iloc[:,]
#     change_points = fetch_change_points(dataset_name)
#     plt.clf()
#     title = f"Original values for {attr} - {dataset_name} dataset"
#     plt.title(title)
#     g = sns.lineplot(x=df.index, y=attr, hue="class", data=df)
#     g.axhline(p_value, ls="--", c="red")
#     if change_points:
#         for change_point in change_points:
#             g.axvline(change_point, ls="--", c="yellow")
#     plt.show()


# Plot original DFs
# for dataset in df_rows.dataset.unique()[:1]:
#     plot_original_df(dataset, species="ae-aegypti-female")


# baseline_dfs = [
#         self.total_df[self.total_df["class"] == species].iloc[
#             : self.window_size,
#         ]
#         for species in self.classes
#     ]
#     self.df_baseline = pd.concat(baseline_dfs)


# dataset_name = "Incremental-gradual (bal.)"
# top5 = [8, 9, 11, 24, 27]
# attrs = [f"Att{x}" for x in top5]

# for index, attr in enumerate(attrs):
#     species = b"ae-aegypti-female"
#     df = load_insect_dataset(insects_datasets[dataset_name]["filename"])
#     df = df[df["class"] == species].iloc[:,]
#     change_points = fetch_change_points(dataset_name)
#     plt.figure(index + 1)
#     plt.clf()
#     title = f"Original values for {attr} - {dataset_name} dataset"
#     plt.title(title)
#     g = sns.lineplot(x=df.index, y=attr, hue="class", data=df)
#     g.axhline(p_value, ls="--", c="red")
#     for change_point in change_points:
#         g.axvline(change_point, ls="--", c="yellow")


# plt.show()


#

from experiment import Experiment

dataset_name = "Incremental-gradual (bal.)"
exp = Experiment(dataset=dataset_name)
exp.prepare_insects_test()

top5 = [8, 9, 11, 24, 27]
attrs = [f"Att{x}" for x in top5]


# Colar o valor do start do p-value no indice original
csv_file, _, plot_file = generate_filenames(dataset_name)
df_analysis = pd.read_csv(csv_file)
df_analysis["action"] = np.where(df_analysis["p_value"] <= p_value, "reject", "accept")
attr_list = list(
    df_analysis[df_analysis["action"] == "accept"].attr.value_counts().to_dict().keys()
)[:5]

change_points = fetch_change_points(dataset_name)


# df_result.index == df_stream.original_index
df_plot = df_analysis[(df_analysis["attr"].isin(attr_list))][
    ["p_value", "attr", "start"]
]
df_plot = df_plot.merge(exp.df_stream, how="inner", left_on="start", right_index=True)
df_plot = df_plot[["p_value", "attr", "start", "original_index", "class"]]


# ####

# df = exp.df_stream
species = b"ae-aegypti-female"

for index, attr in enumerate(attrs):
    df = df_plot[df_plot["class"] == species].iloc[:,]
    change_points = fetch_change_points(dataset_name)
    plt.figure(index + 1)
    plt.clf()
    title = f"P values for {attr} - {dataset_name} dataset"
    plt.title(title)
    g = sns.lineplot(x=df.original_index, y=df.p_value, hue="class", data=df)
    g.axhline(p_value, ls="--", c="red")
    for change_point in change_points:
        g.axvline(change_point, ls="--", c="yellow")


plt.show()