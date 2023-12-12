import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiment import Experiment
from usp_stream_datasets import insects_datasets
from utils import fetch_original_dataframe, generate_experiment_filenames

from config import RESULTS_FOLDER

sns.set(rc={"figure.figsize": (12, 6)})


def fetch_change_points(dataset_name: str) -> list:
    """Fetch change point list for a requested Dataset."""
    return insects_datasets[dataset_name].get("change_point", [])


def fetch_experiment_change_points(dataset_name: str) -> list:
    """Fetch reindexed change points on baseline and stream datasets."""
    exp = Experiment(dataset=dataset_name)
    exp.prepare_insects_test()
    change_points = insects_datasets[dataset_name].get("change_point", [])
    cps = {"baseline": [], "stream": []}

    # Fetch for stream
    for change_point in change_points:
        stream_change_point = exp.df_stream.original_index[
            exp.df_stream.original_index == change_point
        ].index.tolist()
        if len(stream_change_point) > 0:
            cps["stream"].append(
                {"original": change_point, "reindexed": stream_change_point[0]}
            )
    # Fetch for baseline
    for change_point in change_points:
        baseline_change_point = exp.df_baseline.original_index[
            exp.df_baseline.original_index == change_point
        ].index.tolist()
        if len(baseline_change_point) > 0:
            cps["baseline"].append(
                {
                    "original": change_point,
                    "reindexed": baseline_change_point[0],
                }
            )
    return cps


def plot_p_values(series, p_value):
    """Line-plot a series and a horizontal p-value."""
    g = sns.lineplot(data=series)
    g.axhline(p_value, ls="--", c="red")
    plt.figure()
    plt.show()


def plot_dataset_change_points(series, points, p_value):
    """Line-plot a series with vertical change_points."""
    g = sns.lineplot(data=series)
    g.axhline(p_value, ls="--", c="red")
    for point in points:
        g.axvline(point, ls="--", c="yellow")
    plt.figure()
    plt.show()


def plot_original_data(dataset_name: str, species: str = "", attr: str = ""):
    """Create the lineplot of an attribute of a species on a given a dataset."""
    original_dataframe = fetch_original_dataframe(dataset_name)
    change_points = fetch_change_points(dataset_name)
    if not attr:
        print("Please set the attribute to be displayed.")
        return
    if species:
        df = original_dataframe[original_dataframe["class"] == species].iloc[
            :,
        ]
    else:
        df = original_dataframe
    if attr not in df.columns.tolist():
        print("The attribute is not present on the Dataframe.")
        print(f"The available Attrs are: {df.columns.tolist()}")
        return
    plt.clf()
    title = (
        f"Original distribution - {species} - {dataset_name} dataset - {attr}"
    )
    plt.title(title)
    g = sns.lineplot(
        x=df.index,
        y=attr,
        data=df,
    )
    for change_point in change_points:
        g.axvline(change_point, ls="--", c="yellow")
    plt.show()


def fetch_top_n_accepted_attributes(dataset_name, p_value, top_n=10):
    """Fetch top n accepted attributes."""
    csv_file, _, _ = generate_experiment_filenames(
        dataset_name, results_folder=RESULTS_FOLDER
    )
    df_analysis = pd.read_csv(csv_file)
    df_analysis["action"] = np.where(
        df_analysis["p_value"] <= p_value, "reject", "accept"
    )
    top_acceptance_attr_list = list(
        df_analysis[df_analysis["action"] == "accept"]
        .attr.value_counts()
        .to_dict()
        .keys()
    )[:top_n]
    return top_acceptance_attr_list


def plot_multiple_p_values(
    dataset_name, p_value, top_n=10, index="end", save=False
):
    """Given a dataset name and p_value, plot the results of the top 5
    most accepted attributes on that result dataframe."""
    csv_file, _, plot_file = generate_experiment_filenames(
        dataset_name, results_folder=RESULTS_FOLDER
    )
    df_analysis = pd.read_csv(csv_file)
    df_analysis["action"] = np.where(
        df_analysis["p_value"] <= p_value, "reject", "accept"
    )
    attr_list = list(
        df_analysis[df_analysis["action"] == "accept"]
        .attr.value_counts()
        .to_dict()
        .keys()
    )[:top_n]
    df_plot = df_analysis[(df_analysis["attr"].isin(attr_list))][
        ["p_value", "attr", "start", "end"]
    ]
    plt.clf()
    title = f"p-values for {dataset_name} dataset indexed by {index}"
    plt.title(title)
    g = sns.lineplot(x=index, y="p_value", hue="attr", data=df_plot)
    g.axhline(p_value, ls="--", c="red")
    change_points = fetch_experiment_change_points(dataset_name)
    stream_change_points = change_points["stream"]
    if stream_change_points:
        for change_point in stream_change_points:
            g.axvline(change_point["reindexed"], ls="--", c="yellow")
    if save:
        plt.savefig(plot_file)
    plt.show()
