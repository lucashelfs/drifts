import pandas as pd


def create_dummy_bins(baseline, stream, n_bins, bins_origin):

    if bins_origin == "baseline":
        # Create min max
        min_value = baseline.min()
        max_value = baseline.max()

    elif bins_origin == "stream":
        min_value = stream.min()
        max_value = stream.max()

    else:  # both
        min_value = min(baseline.min(), stream.min())
        max_value = max(baseline.max(), stream.max())

    # Add two more bins, using values in ranges (-inf, min) and (max, inf)
    bin_width = (max_value - min_value) / n_bins

    bins = (
        [float("-inf")]
        + [min_value + i * bin_width for i in range(n_bins)]
        + [max_value]
        + [float("inf")]
    )

    return bins


def create_median_bins(baseline, stream, n_bins, bins_origin):

    if bins_origin == "baseline":
        median_value = baseline.median()
        min_value = baseline.min()
        max_value = baseline.max()

    elif bins_origin == "stream":
        median_value = stream.median()
        min_value = stream.min()
        max_value = stream.max()

    else:  # both
        # TODO: check if this is right
        median_value = pd.concat([baseline, stream], ignore_index=True).median()
        min_value = min(baseline.min(), stream.min())
        max_value = max(baseline.max(), stream.max())

    # Calculate bin edges
    bin_width = (max_value - min_value) / n_bins

    bins = (
        [float("-inf")]
        + [median_value - bin_width / 2 + i * bin_width for i in range(n_bins)]
        + [float("inf")]
    )

    return bins


def binner(baseline, stream, **kwargs):

    bins_origin = kwargs.get("bins_origin", False)
    binning_type = kwargs.get("binning_type", False)
    n_bins = kwargs.get("n_bins", False)

    if not n_bins or not binning_type or not bins_origin:
        raise Exception(
            f"Check your binning kwargs! - {n_bins} - {binning_type} - {bins_origin}"
        )

    if binning_type == "dummy":
        bins = create_dummy_bins(baseline, stream, n_bins, bins_origin)
        return bins

    if binning_type == "median":
        bins = create_median_bins(baseline, stream, n_bins, bins_origin)
        return bins

    return False


def frequencier(baseline, stream, **kwargs):

    # The bins will be created accordingly to the kwargs
    bins = binner(baseline, stream, **kwargs)

    if not bins:
        raise Exception("Problem on binning logic!")

    corrected = kwargs.get("corrected", True)

    # Here we apply the bins to the dataframes
    df_bins_baseline = pd.cut(baseline, bins=bins, include_lowest=True)
    df_bins_stream = pd.cut(stream, bins=bins, include_lowest=True)

    baseline_counts = df_bins_baseline.value_counts()
    stream_counts = df_bins_stream.value_counts()

    baseline_normalized_counts = baseline_counts / baseline_counts.sum()
    frequencies_baseline = baseline_normalized_counts

    stream_normalized_counts = stream_counts / stream_counts.sum()
    frequencies_stream = stream_normalized_counts

    all_categories = set(frequencies_baseline.index).union(frequencies_stream.index)
    normalized_counts_baseline = frequencies_baseline.reindex(
        all_categories, fill_value=0
    ).sort_index()
    normalized_counts_stream = frequencies_stream.reindex(
        all_categories, fill_value=0
    ).sort_index()

    if not corrected:
        return normalized_counts_baseline, normalized_counts_stream
    else:
        return dasu_correction(normalized_counts_baseline, normalized_counts_stream)


def dasu_correction(normalized_counts_baseline, normalized_counts_stream):
    total_baseline = normalized_counts_baseline.sum()
    total_stream = normalized_counts_stream.sum()
    dasu_baseline = (normalized_counts_baseline + 0.5) / (
        total_baseline + len(normalized_counts_baseline) / 2
    )
    dasu_stream = (normalized_counts_stream + 0.5) / (
        total_stream + len(normalized_counts_stream) / 2
    )
    return dasu_baseline, dasu_stream
