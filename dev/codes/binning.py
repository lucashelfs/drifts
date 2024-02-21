import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

from scipy.spatial.distance import jensenshannon

N_BINS = 20

# TODO: DRY the methods


def dummy_binning(baseline, stream, bins=N_BINS):
    min_value = min(baseline.min(), stream.min())
    max_value = max(baseline.max(), stream.max())
    bins = np.linspace(min_value, max_value, bins)
    df_bins_baseline = pd.cut(baseline, bins=bins, include_lowest=True)
    df_bins_stream = pd.cut(stream, bins=bins, include_lowest=True)
    return df_bins_baseline, df_bins_stream


def median_binning(
    baseline, stream, median_origin: str = "baseline", n_bins: int = N_BINS
):

    min_value = min(baseline.min(), stream.min())
    max_value = max(baseline.max(), stream.max())

    # Calculate bin edges
    bin_width = (max_value - min_value) / n_bins

    # Calculate the median: what is the best way to choose this?
    if median_origin == "baseline":
        median_value = baseline.median()
    elif median_origin == "stream":
        median_value = stream.median()
    elif median_origin == "both":
        median_value = pd.concat([baseline, stream], ignore_index=True).median()
    else:
        raise Exception("Median origin not defined.")

    bins = (
        [float("-inf")]
        + [median_value - bin_width / 2 + i * bin_width for i in range(n_bins)]
        + [float("inf")]
    )

    df_bins_baseline = pd.cut(baseline, bins=bins, include_lowest=True)
    df_bins_stream = pd.cut(stream, bins=bins, include_lowest=True)
    return df_bins_baseline, df_bins_stream


def simple_binning(baseline, stream, n_bins=N_BINS):

    df_bins_baseline, df_bins_stream = dummy_binning(baseline, stream, n_bins)
    frequencies_baseline = df_bins_baseline.value_counts(normalize=True)[
        df_bins_baseline.unique()
    ]
    frequencies_stream = df_bins_stream.value_counts(normalize=True)[
        df_bins_stream.unique()
    ]

    # Ensure both series have the same index to align them properly
    # -> this is always true because of the linspace...
    all_categories = set(frequencies_baseline.index).union(
        frequencies_stream.index
    )
    normalized_counts_baseline = frequencies_baseline.reindex(
        all_categories, fill_value=0
    )
    normalized_counts_stream = frequencies_stream.reindex(
        all_categories, fill_value=0
    )

    return normalized_counts_baseline, normalized_counts_stream


def plot_binning(
    baseline, stream, binning_type="dummy", median_origin=None, n_bins=N_BINS
):

    if binning_type == "dummy":
        df_bins_baseline, df_bins_stream = dummy_binning(
            baseline, stream, n_bins
        )
    elif binning_type == "median":
        df_bins_baseline, df_bins_stream = median_binning(
            baseline, stream, median_origin=median_origin, n_bins=n_bins
        )

    value_counts1 = df_bins_baseline.value_counts()[df_bins_baseline.unique()]
    value_counts2 = df_bins_stream.value_counts()[df_bins_stream.unique()]

    # Plot the value counts with different colors
    ax = value_counts1.plot(
        kind="bar", color="blue", position=0, width=0.4, label="Baseline"
    )
    value_counts2.plot(
        kind="bar", color="orange", position=1, width=0.4, label="Stream", ax=ax
    )

    # Customize the plot
    plt.xlabel("Bin intervals")
    plt.ylabel("Count")
    plt.title("Value Counts of Bins")
    plt.legend()
    plt.show()


def calculate_kl_with_dummy_bins(baseline, stream, n_bins):

    normalized_baseline, normalized_stream = simple_binning(
        baseline=baseline, stream=stream, n_bins=n_bins
    )

    # Calculate KL divergence
    kl_divergence = entropy(normalized_baseline, normalized_stream)
    return kl_divergence


def calculate_kl_with_median(baseline, stream, median_origin="both", n_bins=5):
    # Calculate KL divergence: median both
    df_bins_baseline, df_bins_stream = median_binning(
        baseline, stream, median_origin=median_origin, n_bins=n_bins
    )

    frequencies_baseline = df_bins_baseline.value_counts(normalize=True)[
        df_bins_baseline.unique()
    ]
    frequencies_stream = df_bins_stream.value_counts(normalize=True)[
        df_bins_stream.unique()
    ]

    all_categories = set(frequencies_baseline.index).union(
        frequencies_stream.index
    )
    normalized_baseline = frequencies_baseline.reindex(
        all_categories, fill_value=0
    )
    normalized_stream = frequencies_stream.reindex(all_categories, fill_value=0)

    kl_divergence = entropy(normalized_baseline, normalized_stream)
    return kl_divergence


##### JENSEN SHANNON


def calculate_js_with_dummy_bins(baseline, stream, n_bins):
    normalized_baseline, normalized_stream = simple_binning(
        baseline=baseline, stream=stream, n_bins=n_bins
    )

    js_distance = jensenshannon(normalized_baseline, normalized_stream)
    return js_distance


def calculate_js_with_median(baseline, stream, median_origin="both", n_bins=5):
    df_bins_baseline, df_bins_stream = median_binning(
        baseline, stream, median_origin=median_origin, n_bins=n_bins
    )

    frequencies_baseline = df_bins_baseline.value_counts(normalize=True)[
        df_bins_baseline.unique()
    ]
    frequencies_stream = df_bins_stream.value_counts(normalize=True)[
        df_bins_stream.unique()
    ]

    all_categories = set(frequencies_baseline.index).union(
        frequencies_stream.index
    )
    normalized_baseline = frequencies_baseline.reindex(
        all_categories, fill_value=0
    )
    normalized_stream = frequencies_stream.reindex(all_categories, fill_value=0)

    js_distance = jensenshannon(normalized_baseline, normalized_stream)
    return js_distance
