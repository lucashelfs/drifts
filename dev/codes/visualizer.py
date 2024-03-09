import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from codes.experiment import Experiment
from codes.usp_stream_datasets import insects_datasets
from codes.utils import fetch_original_dataframe, generate_experiment_filenames

from codes.config import DEFAULT_RESULTS_FOLDER

sns.set(rc={"figure.figsize": (12, 6)})


# TODO: implement a way of visualizing an experiment and DRY on the methods
# TODO: transfer the binning visualizers to this file


def fetch_change_points(dataset_name: str) -> list:
    """Fetch change point list for a requested Dataset."""
    return insects_datasets[dataset_name].get("change_point", [])


def fetch_experiment_change_points(
    results_folder=DEFAULT_RESULTS_FOLDER, **kwargs
) -> list:
    """Fetch reindexed change points on baseline and stream datasets."""
    exp = Experiment(results_folder=results_folder, **kwargs)
    exp.prepare_insects_test()
    change_points = insects_datasets[kwargs.get("dataset")].get("change_point", [])
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
    title = f"Original distribution - {species} - {dataset_name} dataset - {attr}"
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
        dataset_name, results_folder=DEFAULT_RESULTS_FOLDER
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
    dataset_name,
    p_value,
    top_n=10,
    index="end",
    save=False,
    results_folder=DEFAULT_RESULTS_FOLDER,
):
    """Given a dataset name and p_value, plot the results of the top 5
    most accepted attributes on that result dataframe."""

    csv_file, _, plot_file = generate_experiment_filenames(
        dataset_name, results_folder=results_folder
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

    if df_plot.empty:
        print("The dataframe for plotting is empty! The full analysis will be plotted.")
        df_plot = df_analysis

    plt.clf()
    title = f"p-values for {dataset_name} dataset indexed by {index}"
    plt.title(title)
    g = sns.lineplot(x=index, y="p_value", hue="attr", data=df_plot)
    g.axhline(p_value, ls="--", c="red")
    change_points = fetch_experiment_change_points(
        dataset_name, results_folder=results_folder
    )
    stream_change_points = change_points["stream"]
    if stream_change_points:
        for change_point in stream_change_points:
            g.axvline(change_point["reindexed"], ls="--", c="yellow")
    if save:
        plt.savefig(plot_file)
    plt.show()


def plot_kl_values(
    dataset_name,
    attr="Att27",
    index="end",
    save=False,
    results_folder=DEFAULT_RESULTS_FOLDER,
    **kwargs,
):
    """Given a dataset name and attribute, plot the results of the distances
    on that result dataframe."""

    csv_file, _, plot_file = generate_experiment_filenames(
        dataset_name,
        results_folder=results_folder,
    )
    df_analysis = pd.read_csv(csv_file)
    attr_list = [attr]
    df_plot = df_analysis[(df_analysis["attr"].isin(attr_list))][
        ["distance", "attr", "start", "end"]
    ]

    if df_plot.empty:
        print("The dataframe for plotting is empty! The full analysis will be plotted.")
        df_plot = df_analysis

    n_bins = kwargs.get("n_bins", False)

    if not n_bins:
        title = f"Distance for {dataset_name} dataset indexed by {index} - {attr}"
    else:
        title = f"Distance for {dataset_name} dataset indexed by {index} - {attr} - bins: {n_bins}"

    plt.clf()
    plt.title(title)
    g = sns.lineplot(x=index, y="distance", hue="attr", data=df_plot)
    change_points = fetch_experiment_change_points(
        results_folder=results_folder, **kwargs
    )
    stream_change_points = change_points["stream"]
    if stream_change_points:
        for change_point in stream_change_points:
            g.axvline(change_point["reindexed"], ls="--", c="yellow")
    if save:
        plt.savefig(plot_file)
    plt.show()


from codes.base_experiment import BaseExperiment

class InsectsVisualizer:
    def __init__(self, experiment) -> None:
        assert isinstance(experiment, BaseExperiment)
        self.experiment = experiment

    def fetch_experiment_filename(self):
        results_file = self.experiment.dataset_prefix + ".csv"
        metadata_file = self.experiment.dataset_prefix + "metadata.json"
        plot_file = self.experiment.dataset_prefix + ".jpg"
        return results_file, metadata_file, plot_file

    def plot_kl_values(
        self,
        attr="Att27",
        index="end",
        save=False,
        **kwargs,
    ):
        """Given a dataset name and attribute, plot the results of the distances
        on that result dataframe."""

        csv_file, _, plot_file = self.fetch_experiment_filename()
        df_analysis = pd.read_csv(csv_file)
        attr_list = [attr]
        df_plot = df_analysis[(df_analysis["attr"].isin(attr_list))][
            ["distance", "attr", "start", "end"]
        ]

        dataset = kwargs.get("dataset", False)
        if not dataset:
            raise Exception("Missing dataset on experiment visualizer!")

        if df_plot.empty:
            print(
                "The dataframe for plotting is empty! The full analysis will be plotted."
            )
            df_plot = df_analysis

        n_bins = kwargs.get("n_bins", False)

        if not n_bins:
            title = f"Distance for {dataset} dataset indexed by {index} - {attr}"
        else:
            title = f"Distance for {dataset} dataset indexed by {index} - {attr} - bins: {n_bins}"

        plt.clf()
        plt.title(title)
        g = sns.lineplot(x=index, y="distance", hue="attr", data=df_plot)
        change_points = self.fetch_change_points()
        stream_change_points = change_points["stream"]
        if stream_change_points:
            for change_point in stream_change_points:
                g.axvline(change_point["reindexed"], ls="--", c="yellow")
        if save:
            plt.savefig(plot_file)
        plt.show()

    def fetch_change_points(self) -> list:
        """Fetch reindexed change points on baseline and stream datasets."""
        change_points = insects_datasets[self.experiment.dataset].get("change_point", [])
        cps = {"baseline": [], "stream": []}

        # Fetch for stream
        for change_point in change_points:
            stream_change_point = self.experiment.df_stream.original_index[
                self.experiment.df_stream.original_index == change_point
            ].index.tolist()
            if len(stream_change_point) > 0:
                cps["stream"].append(
                    {"original": change_point, "reindexed": stream_change_point[0]}
                )
        # Fetch for baseline
        for change_point in change_points:
            baseline_change_point = self.experiment.df_baseline.original_index[
                self.experiment.df_baseline.original_index == change_point
            ].index.tolist()
            if len(baseline_change_point) > 0:
                cps["baseline"].append(
                    {
                        "original": change_point,
                        "reindexed": baseline_change_point[0],
                    }
                )
        return cps
