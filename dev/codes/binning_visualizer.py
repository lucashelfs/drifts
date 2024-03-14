import imageio
import matplotlib.pyplot as plt

from binning import dummy_binning, median_binning


def plot_binning(baseline, stream, binning_type="dummy", median_origin=None, **kwargs):
    if binning_type == "dummy":
        df_bins_baseline, df_bins_stream = dummy_binning(
            baseline, stream, n_bins=kwargs.get("n_bins")
        )
    elif binning_type == "median":
        df_bins_baseline, df_bins_stream = median_binning(
            baseline, stream, median_origin=median_origin, n_bins=kwargs.get("n_bins")
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


def create_binning_frame(baseline, stream, step, binning_type="dummy", **kwargs):
    """
    References:
    - https://freedium.cfd/https://towardsdatascience.com/how-to-create-a-gif-from-matplotlib-plots-in-python-6bec6c0c952c
    """
    if binning_type == "dummy":
        df_bins_baseline, df_bins_stream = dummy_binning(
            baseline, stream, n_bins=kwargs.get("n_bins")
        )
        binning_str = ""
    elif binning_type == "median":
        median_origin = kwargs.get("median_origin")
        binning_str = f" - Median - {median_origin}"
        df_bins_baseline, df_bins_stream = median_binning(
            baseline,
            stream,
            median_origin=kwargs.get("median_origin"),
            n_bins=kwargs.get("n_bins"),
        )

    value_counts1 = df_bins_baseline.value_counts()[df_bins_baseline.unique()]
    value_counts2 = df_bins_stream.value_counts()[df_bins_stream.unique()]

    # Plot the value counts with different colors
    ax = value_counts1.plot(
        kind="barh", color="blue", position=0, width=0.4, label="Baseline"
    )
    value_counts2.plot(
        kind="barh", color="orange", position=1, width=0.4, label="Stream", ax=ax
    )

    # Customize the plot
    n_bins = kwargs.get("n_bins")
    plt.ylabel("Bin intervals")
    plt.xlabel("Count")
    plt.title(f"Counts of {n_bins} Bins {binning_str}")
    plt.legend()
    plt.savefig(f"./img/img_{step}.png", transparent=False, facecolor="white")
    plt.close()


def create_binning_timeline(df_baseline, df_stream, **kwargs):
    frames = []

    # Jump window in batches with the size of the stream
    window_size = len(df_baseline)

    # TODO: fetch the steps of gif timeline with kwargs
    starts = list(range(0, len(df_stream) - window_size - 1, 1000))

    for start_idx in starts:
        start = start_idx + 1
        end = start + window_size
        stream = df_stream[start:end]
        baseline = df_baseline[kwargs.get("attr")]
        stream = df_stream[kwargs.get("attr")][start:end]
        create_binning_frame(baseline, stream, start_idx, **kwargs)

    for t in starts:
        image = imageio.v2.imread(f"./img/img_{t}.png")
        frames.append(image)

    imageio.mimsave(
        f'./gifs/{kwargs.get("n_bins")}_bins_{kwargs.get("attr")}.gif', frames, fps=5
    )
