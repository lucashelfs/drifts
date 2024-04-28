import imageio
import matplotlib.pyplot as plt

from codes.binning import dummy_binning, median_binning, dummy_binning_v2


def plot_binning(baseline, stream, binning_type="dummy", bins_origin=None, **kwargs):
    if binning_type == "dummy":
        df_bins_baseline, df_bins_stream = dummy_binning(
            baseline, stream, n_bins=kwargs.get("n_bins")
        )
    elif binning_type == "median":
        df_bins_baseline, df_bins_stream = median_binning(
            baseline, stream, bins_origin=bins_origin, n_bins=kwargs.get("n_bins")
        )

    baseline_counts = df_bins_baseline.value_counts()
    stream_counts = df_bins_stream.value_counts()

    all_categories = set(baseline_counts.index).union(stream_counts.index)

    # Plot the value counts with different colors
    ax = (
        baseline_counts.reindex(all_categories, fill_value=0)
        .sort_index()
        .plot(kind="barh", color="blue", position=0, width=0.4, label="Baseline")
    )
    stream_counts.reindex(all_categories, fill_value=0).sort_index().plot(
        kind="barh", color="orange", position=1, width=0.4, label="Stream", ax=ax
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
        df_bins_baseline, df_bins_stream = dummy_binning_v2(
            baseline, stream, n_bins=kwargs.get("n_bins")
        )
        binning_str = ""

    elif binning_type == "median":
        bins_origin = kwargs.get("bins_origin")
        binning_str = f" - Median - {bins_origin}"
        df_bins_baseline, df_bins_stream = median_binning(
            baseline,
            stream,
            bins_origin=kwargs.get("bins_origin"),
            n_bins=kwargs.get("n_bins"),
        )

    baseline_counts = df_bins_baseline.value_counts()
    stream_counts = df_bins_stream.value_counts()

    all_categories = set(baseline_counts.index).union(stream_counts.index)

    # Plot the value counts with different colors
    ax = (
        baseline_counts.reindex(all_categories, fill_value=0)
        .sort_index()
        .plot(kind="barh", color="blue", position=0, width=0.4, label="Baseline")
    )
    stream_counts.reindex(all_categories, fill_value=0).sort_index().plot(
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


def create_binning_timeline(df_baseline, df_stream, filename, **kwargs):
    frames = []

    # Jump window in stepss
    window_size = len(df_baseline)

    gif_step = kwargs.get("gif_frames_step", 1000)
    starts = list(range(0, len(df_stream) - window_size - 1, gif_step))

    for start_idx in starts:
        start = start_idx + 1
        end = start + window_size
        stream = df_stream[start:end]
        baseline = df_baseline[kwargs.get("attr", "value")]
        stream = df_stream[kwargs.get("attr", "value")][start:end]
        create_binning_frame(baseline, stream, start_idx, **kwargs)

    for t in starts:
        image = imageio.v2.imread(f"./img/img_{t}.png")
        frames.append(image)

    if not filename:
        filename = (
            f'./gifs/{kwargs.get("n_bins")}_bins_{kwargs.get("attr", "value")}.gif'
        )

    imageio.mimsave(
        filename,
        frames,
        fps=5,
    )
