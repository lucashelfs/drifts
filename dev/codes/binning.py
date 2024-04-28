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


def create_dummy_bins(baseline, n_bins):
    # Create min max
    min_value = baseline.min()
    max_value = baseline.max()

    # Add two more bins, using values in ranges (-inf, min) and (max, inf)
    bin_width = (max_value - min_value) / n_bins

    bins = (
        [float("-inf")]
        + [min_value + i * bin_width for i in range(n_bins)]
        + [max_value]
        + [float("inf")]
    )

    return bins


def dummy_binning_v2(baseline, stream, n_bins=N_BINS):
    bins = create_dummy_bins(baseline, n_bins)
    df_bins_baseline = pd.cut(baseline, bins=bins, include_lowest=True)
    df_bins_stream = pd.cut(stream, bins=bins, include_lowest=True)
    return df_bins_baseline, df_bins_stream


def median_binning(baseline, stream, bins_origin: str = "None", n_bins: int = N_BINS):
    min_value = min(baseline.min(), stream.min())
    max_value = max(baseline.max(), stream.max())

    # Calculate bin edges
    bin_width = (max_value - min_value) / n_bins

    # Calculate the median: what is the best way to choose this?
    if bins_origin == "baseline":
        median_value = baseline.median()
    elif bins_origin == "stream":
        median_value = stream.median()
    elif bins_origin == "both":
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


def simple_binning_old(baseline, stream, n_bins=N_BINS):
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


# def simple_binning_dasu_corrected(baseline, stream, n_bins=N_BINS):
# @staticmethod
# def _distn_from_counts(counts):
#     """Calculate an empirical distribution across the alphabet defined by
#     the bins (here, indices in the counts array), using Dasu's correction.

#     Args:
#         counts (numpy.array): array of counts

#     Returns:
#         numpy.array: array of frequencies
#     """
#     total = np.sum(counts)
#     hist = np.array(counts) + 0.5
#     hist = hist / (total + len(hist) / 2)
#     return hist


def simple_binning_dasu_corrected(baseline, stream, n_bins=N_BINS):
    df_bins_baseline, df_bins_stream = dummy_binning_v2(baseline, stream, n_bins)
    baseline_counts = df_bins_baseline.value_counts()
    baseline_normalized_counts = baseline_counts / baseline_counts.sum()
    frequencies_baseline = baseline_normalized_counts
    stream_counts = df_bins_stream.value_counts()
    stream_normalized_counts = stream_counts / stream_counts.sum()
    frequencies_stream = stream_normalized_counts
    all_categories = set(frequencies_baseline.index).union(frequencies_stream.index)
    normalized_counts_baseline = frequencies_baseline.reindex(
        all_categories, fill_value=0
    ).sort_index()
    normalized_counts_stream = frequencies_stream.reindex(
        all_categories, fill_value=0
    ).sort_index()
    total_baseline = normalized_counts_baseline.sum()
    total_stream = normalized_counts_stream.sum()
    dasu_baseline = (normalized_counts_baseline + 0.5) / (
        total_baseline + len(normalized_counts_baseline) / 2
    )
    dasu_stream = (normalized_counts_stream + 0.5) / (
        total_stream + len(normalized_counts_stream) / 2
    )
    return dasu_baseline, dasu_stream


def calculate_kl_with_dummy_bins(baseline, stream, **kwargs):

    # normalized_baseline, normalized_stream = simple_binning_dasu_corrected(
    #     baseline=baseline, stream=stream, n_bins=kwargs.get("n_bins", 5)
    # )

    from codes.binning_v2 import frequencier

    normalized_baseline, normalized_stream = frequencier(baseline, stream, **kwargs)

    # Calculate KL divergence
    kl_divergence = entropy(normalized_baseline, normalized_stream)
    return kl_divergence


def calculate_kl_with_median(baseline, stream, **kwargs):
    # # Calculate KL divergence: median both
    # df_bins_baseline, df_bins_stream = median_binning(
    #     baseline, stream, bins_origin=bins_origin, n_bins=n_bins
    # )

    # frequencies_baseline = df_bins_baseline.value_counts(normalize=True)[
    #     df_bins_baseline.unique()
    # ]
    # frequencies_stream = df_bins_stream.value_counts(normalize=True)[
    #     df_bins_stream.unique()
    # ]

    # all_categories = set(frequencies_baseline.index).union(frequencies_stream.index)

    # normalized_baseline = frequencies_baseline.reindex(all_categories, fill_value=0)
    # normalized_stream = frequencies_stream.reindex(all_categories, fill_value=0)

    from codes.binning_v2 import frequencier

    normalized_baseline, normalized_stream = frequencier(baseline, stream, **kwargs)

    kl_divergence = entropy(normalized_baseline, normalized_stream)
    return kl_divergence


def calculate_kl(baseline, stream, **kwargs):

    from codes.binning_v2 import frequencier

    normalized_baseline, normalized_stream = frequencier(baseline, stream, **kwargs)

    kl_divergence = entropy(normalized_baseline, normalized_stream)
    return kl_divergence


# JENSEN SHANNON
def calculate_js(baseline, stream, **kwargs):
    from codes.binning_v2 import frequencier

    normalized_baseline, normalized_stream = frequencier(baseline, stream, **kwargs)
    js_distance = jensenshannon(normalized_baseline, normalized_stream)
    return js_distance


def calculate_js_with_dummy_bins(baseline, stream, n_bins):
    normalized_baseline, normalized_stream = simple_binning_old(
        baseline=baseline, stream=stream, n_bins=n_bins
    )

    js_distance = jensenshannon(normalized_baseline, normalized_stream)
    return js_distance


def calculate_js_with_median(baseline, stream, bins_origin="both", n_bins=5):
    df_bins_baseline, df_bins_stream = median_binning(
        baseline, stream, bins_origin=bins_origin, n_bins=n_bins
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


# HELLINGER DISTANCE
from scipy.linalg import norm

_SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64


def hellinger(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def calculate_hellinger(baseline, stream, **kwargs):
    from codes.binning_v2 import frequencier

    normalized_baseline, normalized_stream = frequencier(baseline, stream, **kwargs)
    return hellinger(normalized_baseline, normalized_stream)


def calculate_hellinger_with_dummy_bins(baseline, stream, n_bins):
    normalized_baseline, normalized_stream = simple_binning_old(
        baseline=baseline, stream=stream, n_bins=n_bins
    )
    return hellinger(normalized_baseline, normalized_stream)


def calculate_hellinger_with_median(baseline, stream, bins_origin="both", n_bins=5):
    df_bins_baseline, df_bins_stream = median_binning(
        baseline, stream, bins_origin=bins_origin, n_bins=n_bins
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

    return hellinger(normalized_baseline, normalized_stream)
