from dev.config import insects_datasets, load_insect_dataset
from dev.simple_ksddm import SimpleKSDDM

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dev.common import load_and_prepare_dataset


def define_batches(X, batch_size):
    X["Batch"] = (X.index // batch_size) + 1
    return X


def initialize_ksddm(reference, mean_threshold=0.05):
    ksddm = SimpleKSDDM(mean_threshold=mean_threshold)
    ksddm.set_reference(reference)
    return ksddm


def process_batches(ksddm, X, reference_batch):
    batches = X[X.Batch != reference_batch].Batch.unique()
    heatmap_data = pd.DataFrame(columns=batches)
    diff_heatmap_data = pd.DataFrame(columns=batches)
    detected_drift = []

    for batch, subset_data in X[X.Batch != reference_batch].groupby(
        "Batch"
    ):
        ksddm.update(subset_data.iloc[:, :-1])
        heatmap_data[batch] = ksddm.p_values
        diff_heatmap_data[batch] = ksddm.p_value_diff
        detected_drift.append(ksddm.drift_state)

    return heatmap_data, diff_heatmap_data, detected_drift


def add_drift_spans(ax, plot_data):
    for i, t in enumerate(
        plot_data.loc[plot_data["Detected Drift"] == "drift"]["Batch"]
    ):
        ax.axvspan(
            t - 0.2,
            t + 0.2,
            alpha=0.5,
            color="red",
            label=("Drift Detected" if i == 0 else None),
        )


def plot_statistics(
    plot_data,
    dataset,
    batch_size,
    batches,
    change_points,
    epsilons,
    thresholds,
    detected_drift,
):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot("Batch", "minimum_values", data=plot_data, label="minimum_values", marker=".")
    ax.plot("Batch", "maximum_values", data=plot_data, label="maximum_values", marker=".")
    ax.grid(False, axis="x")
    ax.set_xticks(batches)
    ax.set_xticklabels(batches, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_title(
        f"KSDDM minimum_values and max p-values for dataset {dataset} - {batch_size}",
        fontsize=22,
        pad=50,
    )
    fig.subplots_adjust(top=0.75)
    ax.set_ylabel("p-values mean Distance Values", fontsize=18)
    ax.set_xlabel("Batch", fontsize=18)
    ax.set_ylim(
        [
            min(min(epsilons), min(thresholds)) - 0.02,
            max(max(epsilons), max(thresholds)) + 0.02,
        ]
    )

    for cp in change_points:
        batch_number = cp // batch_size
        ax.axvline(x=batch_number, color="red", linestyle="--", linewidth=1.5)
        ax.text(
            batch_number,
            max(max(epsilons), max(thresholds)) + 0.01,
            f"CP: {cp}",
            ha="center",
            fontsize=10,
            rotation=45,
        )

    add_drift_spans(ax, plot_data)

    ax.legend()
    ax.axhline(y=0, color="orange", linestyle="dashed")
    plt.show()


def plot_values(plot_data, batches_list, dataset_name, batch_size, change_points, value_to_plot=""):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot("Batch", value_to_plot, data=plot_data, label=value_to_plot, marker=".")
    ax.grid(False, axis="x")
    ax.set_xticks(batches_list)
    ax.set_xticklabels(batches_list, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_title(
        f"KSDDM {value_to_plot} p-values for dataset {dataset_name} - {batch_size}",
        fontsize=22,
        pad=50,
    )
    fig.subplots_adjust(top=0.75)
    ax.set_ylabel(f"{value_to_plot} of p-values", fontsize=18)
    ax.set_xlabel("Batch", fontsize=18)
    ax.set_ylim(
        [
            plot_data[value_to_plot].min() - 0.02,
            plot_data[value_to_plot].max() + 0.02,
        ]
    )

    for cp in change_points:
        batch_number = cp // batch_size
        ax.axvline(x=batch_number, color="red", linestyle="--", linewidth=1.5)
        ax.text(
            batch_number,
            plot_data[value_to_plot].max() + 0.01,
            f"CP: {cp}",
            ha="center",
            fontsize=10,
            rotation=45,
        )

    add_drift_spans(ax, plot_data)

    ax.legend()
    ax.axhline(y=0, color="orange", linestyle="dashed")
    plt.show()


def plot_p_values_delta(
    plot_data, dataset, batch_size, batches, change_points, p_values_delta
):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(
        "Batch", "p-values_delta", data=plot_data, label="p-values Delta", marker="."
    )
    ax.grid(False, axis="x")
    ax.set_xticks(batches)
    ax.set_xticklabels(batches, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_title(
        f"KSDDM p-values Delta for dataset {dataset} - {batch_size}",
        fontsize=22,
        pad=50,
    )
    fig.subplots_adjust(top=0.75)
    ax.set_ylabel("p-values Delta", fontsize=18)
    ax.set_xlabel("Batch", fontsize=18)
    ax.set_ylim([min(p_values_delta) - 0.02, max(p_values_delta) + 0.02])

    for cp in change_points:
        batch_number = cp // batch_size
        ax.axvline(x=batch_number, color="red", linestyle="--", linewidth=1.5)
        ax.text(
            batch_number,
            max(p_values_delta) + 0.01,
            f"CP: {cp}",
            ha="center",
            fontsize=10,
            rotation=45,
        )

    add_drift_spans(ax, plot_data)

    ax.legend()
    ax.axhline(y=0, color="orange", linestyle="dashed")
    plt.show()


def plot_heatmap(heatmap_data, dataset, batch_size, change_points, suffix=""):
    sns.set(rc={"figure.figsize": (15, 8)})
    grid_kws = {"height_ratios": (0.9, 0.05), "hspace": 0.3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    coloring = sns.cubehelix_palette(start=0.8, rot=-0.5, as_cmap=True, reverse=True)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=coloring,
        vmax=0.05 if "Chunked" in suffix else 1,
        xticklabels=heatmap_data.columns,
        yticklabels=heatmap_data.index,
        linewidths=0.5,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
    )

    # Add red lines for change points
    for i, cp in enumerate(change_points):
        batch_number = cp // batch_size
        ax.axvline(
            x=batch_number + 0.5,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Change Point" if i == 0 else "",
        )

    # Add green lines for batches where the mean of the values is less than 0.05
    for i, batch in enumerate(heatmap_data.columns):
        if heatmap_data[batch].mean() < 0.05:
            ax.axvline(
                x=heatmap_data.columns.get_loc(batch) + 0.5,
                color="green",
                linestyle="-",
                linewidth=1.5,
                label="Mean < 0.05" if i == 0 else "",
            )

    ax.set_title(f"KS P-value Heatmap for dataset {dataset} - {suffix} - {batch_size}")
    ax.set(xlabel="Batch", ylabel="Features")
    ax.collections[0].colorbar.set_label("Difference in p-values mean distance")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    plt.show()


def run_ksddm(dataset="Abrupt (bal.)"):
    imgs_folder_prefix = "./test_ksddm/"
    # dataset = "Incremental (bal.)"
    batch_size = 1000
    reference_batch = 1

    X, dataset_filename_str = load_and_prepare_dataset(dataset)
    X = define_batches(X, batch_size)

    reference = X[X.Batch == reference_batch].iloc[:, :-1]
    all_test = X[X.Batch != reference_batch]

    ksddm = initialize_ksddm(reference)
    heatmap_data, diff_heatmap_data, detected_drift = process_batches(
        ksddm, X, reference_batch
    )

    minimum_values = []
    maximum_values = []
    mean_values = []

    for feature in heatmap_data.columns:
        minimum_values.append(heatmap_data[feature].min())
        maximum_values.append(heatmap_data[feature].max())
        mean_values.append(heatmap_data[feature].mean())

    minimum_values_series = pd.Series(minimum_values)
    maximum_values_series = pd.Series(maximum_values)
    mean_values_series = pd.Series(mean_values)

    plot_data = pd.DataFrame(
        {
            "Batch": all_test.Batch.unique(),
            "Detected Drift": ksddm.drift_state,
            "minimum_values": minimum_values_series,
            "maximum_values": maximum_values_series,
            "mean_values": mean_values_series
        }
    )

    batches_list = all_test.Batch.unique()
    change_points = insects_datasets[dataset]["change_point"]

    plot_values(plot_data, batches_list, dataset, batch_size, change_points, value_to_plot="minimum_values")
    plot_values(plot_data, batches_list, dataset, batch_size, change_points, value_to_plot="maximum_values")
    plot_values(plot_data, batches_list, dataset, batch_size, change_points, value_to_plot="mean_values")

    print(plot_data["Detected Drift"])

    # plot_heatmap(
    #     heatmap_data, dataset, batch_size, insects_datasets[dataset]["change_point"]
    # )

    # plot_heatmap(
    #     heatmap_data,
    #     dataset,
    #     batch_size,
    #     insects_datasets[dataset]["change_point"],
    #     "Chunked",
    # )

    # plot_heatmap(
    #     diff_heatmap_data,
    #     dataset,
    #     batch_size,
    #     insects_datasets[dataset]["change_point"],
    #     suffix="DIFF",
    # )

    # plot_heatmap(
    #     diff_heatmap_data,
    #     dataset,
    #     batch_size,
    #     insects_datasets[dataset]["change_point"],
    #     suffix="DIFF Chunked",
    # )

    # plot_heatmap(
    #     min_heatmap_data,
    #     dataset,
    #     batch_size,
    #     insects_datasets[dataset]["change_point"],
    #     suffix="minimum_values",
    # )

    # plot_heatmap(
    #     max_heatmap_data,
    #     dataset,
    #     batch_size,
    #     insects_datasets[dataset]["change_point"],
    #     suffix="maximum_values",
    # )


# from dev.ddm import fetch_ksddm_drifts
#
#
# if __name__ == "__main__":
#     print(fetch_ksddm_drifts())
