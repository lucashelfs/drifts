import os
from codes.insects_experiment import InsectsExperiment, InsectsVisualizer


if __name__ == "__main__":

    n_bins = 5
    dataset_name = "Incremental (bal.)"
    attr = "Att31"
    test_type = "ks"
    binning_type = "median"
    median_origin = "both"

    results_folder = os.getcwd() + f"/TEST-results_of_{test_type}_{n_bins}_{attr}/"

    config = dict(
        batches=False,
        dataset=dataset_name,
        attr=attr,
        results_folder=results_folder,
        test_type=test_type,
        n_bins=n_bins,
        binning_type=binning_type,
        median_origin=median_origin,
    )

    # Set the experiment
    ie = InsectsExperiment(**config)

    # Run the experiment
    ie.run_test()

    # Set a visualizer for the experiment result
    iv = InsectsVisualizer(ie)

    # Plot the obtained results
    iv.plot_result_values(**config)

    # # Plot original data for the attr for example
    # iv.plot_original_data(**config)

    # Generate GIF of stream binning
    # create_binning_timeline(exp.df_baseline, exp.df_stream, **config)

    # TODO: improve below
    # Plot first baseline bin
    # WINDOW_SIZE = len(exp.df_baseline[attr])
    # plot_binning(exp.df_baseline[attr],
    #              exp.df_stream[attr][:WINDOW_SIZE],
    #              **config)
