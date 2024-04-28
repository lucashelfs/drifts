import os
import numpy as np
from codes.insects_experiment import InsectsExperiment, InsectsVisualizer
from codes.binning_visualizer import create_binning_timeline

from codes.usp_stream_datasets import insects_datasets


if __name__ == "__main__":

    n_bins = 5
    dataset_name = "Incremental (bal.)"
    attr = "Att31"
    # window_size = 1000
    window_size = None
    corrected = True
    test_type = "kl"
    binning_type = "dummy"
    bins_origin = "baseline"

    # When loooping for all...
    # for dataset_name in insects_datasets.keys():
    # KL/KS
    # test_types = ["kl", "ks"]
    # binning_types = ["dummy"]
    # bins_origins = ["baseline"]

    # for dataset_name in insects_datasets.keys():

    #     if dataset_name != "Incremental (bal.)":

    dataset_filename_str = (
        dataset_name.lower()
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
    )

    results_folder = (
        os.getcwd()
        + f"/results_insects/{dataset_filename_str}_{test_type}_{binning_type}_{n_bins}"
    )

    if window_size:
        results_folder += f"_window_size_{window_size}"

    if attr:
        results_folder += f"_{attr}"

    results_folder += "/"

    config = dict(
        batches=False,
        dataset=dataset_name,
        attr=attr,
        results_folder=results_folder,
        test_type=test_type,
        n_bins=n_bins,
        binning_type=binning_type,
        bins_origin=bins_origin,
        window_size=window_size,
    )

    ie = InsectsExperiment(**config)

    # Run the experiment
    ie.run_test()

    # # TODO: fetch the number of bins in a dynamic way
    # # Set the experiment just for fetching the default window size
    # default_dataset_window_size = ie.window_size

    # n_bins_list = [x for x in range(5, int(np.sqrt(default_dataset_window_size)), 5)]
    # if int(np.sqrt(window_size)) not in n_bins_list:
    #     n_bins_list.append(int(np.sqrt(window_size)))

    # # ALL THE FORS GO HERE
    # # fetch new config and run...

    # config = dict(
    #     batches=False,
    #     dataset=dataset_name,
    #     # attr=attr,
    #     results_folder=results_folder,
    #     test_type=test_type,
    #     n_bins=n_bins,
    #     binning_type=binning_type,
    #     bins_origin=bins_origin,
    #     window_size=window_size,
    # )

    # # Set the experiment just for fetching the default window size
    # ie = InsectsExperiment(**config)

    # # Run the experiment
    # ie.run_test()

    # # Set a visualizer for the experiment result
    # iv = InsectsVisualizer(ie)

    # # # Plot the obtained results
    # iv.plot_result_values(**config)

    # # Plot original data for the attr for example
    # iv.plot_original_data(**config)

    # Generate GIF of stream binning: how do this works for each attr??
    # create_binning_timeline(ie.df_baseline, ie.df_stream, **config)
