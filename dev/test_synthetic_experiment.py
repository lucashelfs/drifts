import os
from codes.synthetic_experiment import SyntheticExperiment, SynthethicVisualizer
from codes.binning_visualizer import create_binning_timeline

import numpy as np

# synthetic_params = [
#     random_state.normal(0.5, 0.05, 1500),
#     random_state.normal(0.5, 0.1, 1000),
#     random_state.normal(0.5, 0.2, 1000),
# ]

## KS config
# test_type = "ks"
# window_size = 500
# results_folder = os.getcwd() + f"/TEST-SYNTHETIC-results_of_{test_type}/"

# config = dict(
#     batches=False,
#     results_folder=results_folder,
#     test_type=test_type,
#     window_size=window_size,
# )


# Synthetic data params
random_state = np.random.RandomState(seed=42)
visualize_stream = True

synthetic_params = [
    random_state.normal(0.5, 0.1, 2500),
    random_state.normal(0.7, 0.15, 2000),
    random_state.normal(0.9, 0.2, 2000),
    # random_state.normal(0.7, 0.15, 2000),
    # random_state.normal(0.5, 0.2, 2500),
    # random_state.normal(0.5, 0.1, 2500),
]


if __name__ == "__main__":

    baseline_size = 500
    window_size = 500

    # JS/KL/Hellinger config
    test_types = ["kl", "js", "hellinger"]
    binning_types = ["dummy", "median"]
    bins_origins = ["baseline", "stream", "both"]

    corrected = True

    # n_bins_list = [5]
    n_bins_list = [x for x in range(5, int(np.sqrt(window_size)), 5)]
    if int(np.sqrt(window_size)) not in n_bins_list:
        n_bins_list.append(int(np.sqrt(window_size)))

    ## Loop for all binning based...
    # for test_type in test_types:
    #     for binning_type in binning_types:
    #         for bins_origin in bins_origins:
    #             for n_bins in n_bins_list:

    # Single run for KS
    test_type = "ks"
    n_bins = 0
    binning_type = False
    bins_origin = None

    folder_name = f"/results_synthetic/base_{test_type}"

    if test_type != "ks":
        folder_name += f"_{n_bins}_bins_{binning_type}_{bins_origin}"

    if not corrected:
        folder_name += "_uncorrected/"
    else:
        folder_name += "/"

    results_folder = os.getcwd() + folder_name

    config = dict(
        batches=False,
        results_folder=results_folder,
        test_type=test_type,
        window_size=window_size,
        baseline_size=baseline_size,
        n_bins=n_bins,
        visualize_stream=visualize_stream,
        distributions=synthetic_params,
        bins_origin=bins_origin,
        binning_type=binning_type,
        gif_frames_step=250,
        corrected=corrected,
        savefig=True,
        displayfig=False,
    )

    # Set the experiment
    se = SyntheticExperiment(**config)
    se.run_test()

    # Set a visualizer for the experiment result
    sv = SynthethicVisualizer(se)

    # Plot the obtained results
    sv.plot_result_values(**config)

    # Create gif
    _, _, _, binning_timeline_filename = sv.fetch_experiment_filename()

    # TODO: incorporate the binning timeline on the base visualizer in the future.
    # TODO: fixar o eixo X
    # TODO: check gifs from median and both
    if test_type != "ks":
        create_binning_timeline(
            se.df_baseline,
            se.df_stream,
            filename=binning_timeline_filename,
            **config,
        )
