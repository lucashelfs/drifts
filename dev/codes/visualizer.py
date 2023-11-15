import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from usp_stream_datasets import insects_datasets, load_insect_dataset

sns.set(rc={"figure.figsize": (12, 6)})
tested_datasets = []

p_value = 0.05


def generate_filenames(dataset: str):
    results_folder = "results/"
    dataset_prefix = results_folder + dataset.lower().replace(".", "").replace(
        "(", ""
    ).replace(")", "").replace(" ", "-")
    csv_file = dataset_prefix + ".csv"
    metadata_file = dataset_prefix + "_metadata.json"
    plot_file = dataset_prefix + ".jpg"
    return csv_file, metadata_file, plot_file


def open_metadata_file(filename):
    f = open(filename, "r")
    return json.loads(f.read())


def plot_p_values(series, p_value):
    g = sns.lineplot(data=series)
    g.axhline(p_value, ls="--", c="red")
    # plt.figure().set_figwidth(WIDTH)
    plt.figure()
    plt.show()


def plot_dataset_change_points(series, points, p_value):
    g = sns.lineplot(data=series)
    g.axhline(p_value, ls="--", c="red")
    for point in points:
        g.axvline(point, ls="--", c="yellow")
    plt.figure()
    plt.show()


def fetch_change_points(dataset_name: str):
    return insects_datasets[dataset_name].get("change_point", None)


def plot_multiple_p_values(dataset_name, p_value, save=False):
    csv_file, metadata_file, plot_file = generate_filenames(dataset_name)
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
            )  # we need to slide this to the front, but how?
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


def plot_original_df(dataset_name, attr="Att1", species=None):
    """Load dataframe from the insects datasets."""
    df = load_insect_dataset(insects_datasets[dataset_name]["filename"])
    if species:
        df = df[df["class"] == species].iloc[
            :,
        ]
    change_points = fetch_change_points(dataset_name)
    plt.clf()
    title = f"Original values for {attr} - {dataset_name} dataset"
    plt.title(title)
    g = sns.lineplot(x=df.index, y=attr, hue="class", data=df)
    g.axhline(p_value, ls="--", c="red")
    if change_points:
        for change_point in change_points:
            g.axvline(change_point, ls="--", c="yellow")
    plt.show()


# Plot original DFs
for dataset in df_rows.dataset.unique()[:1]:
    plot_original_df(dataset, species="ae-aegypti-female")


# baseline_dfs = [
#         self.total_df[self.total_df["class"] == species].iloc[
#             : self.window_size,
#         ]
#         for species in self.classes
#     ]
#     self.df_baseline = pd.concat(baseline_dfs)


dataset_name = "Incremental-gradual (bal.)"
attr = "Att1"
species = b"ae-aegypti-female"
df = load_insect_dataset(insects_datasets[dataset_name]["filename"])
df = df[df["class"] == species].iloc[
    :,
]
change_points = fetch_change_points(dataset_name)
plt.clf()
title = f"Original values for {attr} - {dataset_name} dataset"
plt.title(title)
g = sns.lineplot(x=df.index, y=attr, hue="class", data=df)
g.axhline(p_value, ls="--", c="red")


if change_points:
    for change_point in change_points:
        g.axvline(change_point, ls="--", c="yellow")


plt.show()
