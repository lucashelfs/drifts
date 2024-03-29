import numpy as np
import pandas as pd
from scipy.stats import entropy

from scipy.spatial.distance import jensenshannon

N_BINS = 20


"""
File with the distances that use binning strategies to be computed.

References:
- https://freedium.cfd/https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254
- https://freedium.cfd/https://towardsdatascience.com/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6
"""


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
    all_categories = set(frequencies_baseline.index).union(frequencies_stream.index)
    normalized_counts_baseline = frequencies_baseline.reindex(
        all_categories, fill_value=0
    )
    normalized_counts_stream = frequencies_stream.reindex(all_categories, fill_value=0)

    return normalized_counts_baseline, normalized_counts_stream


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

    all_categories = set(frequencies_baseline.index).union(frequencies_stream.index)
    normalized_baseline = frequencies_baseline.reindex(all_categories, fill_value=0)
    normalized_stream = frequencies_stream.reindex(all_categories, fill_value=0)

    kl_divergence = entropy(normalized_baseline, normalized_stream)
    return kl_divergence


# JENSEN SHANNON
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

    all_categories = set(frequencies_baseline.index).union(frequencies_stream.index)
    normalized_baseline = frequencies_baseline.reindex(all_categories, fill_value=0)
    normalized_stream = frequencies_stream.reindex(all_categories, fill_value=0)

    js_distance = jensenshannon(normalized_baseline, normalized_stream)
    return js_distance
