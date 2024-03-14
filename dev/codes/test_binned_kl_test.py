import os
from experiment import Experiment
from visualizer import plot_kl_values
from binning_visualizer import plot_binning, create_binning_timeline


# TEST IF IT IS WORKING FOR DUMMY TESTS>>>>

dataset_name = "Incremental (bal.)"
attr = "Att31"
test_type = "kl_dummy"
binning_type = "median"
median_origin = "both"
bins = [5, 10, 25]


for n_bins in bins:
    results_folder = os.getcwd() + f"/results_of_{test_type}_{n_bins}_{attr}/"
    config = dict(
        batches=False,
        dataset_name=dataset_name,
        dataset=dataset_name,
        attr=attr,
        results_folder=results_folder,
        data_source="insects",
        test_type=test_type,
        n_bins=n_bins,
        binning_type=binning_type,
        median_origin=median_origin,
    )
    # Run experiment for single attr
    exp = Experiment(**config)
    exp.prepare_insects_test()
    # exp.run_insects_test()
    plot_kl_values(**config)

    # Generate GIF of stream binning
    # create_binning_timeline(exp.df_baseline, exp.df_stream, **config)


# # Plot values for that attribute
# plot_kl_values(**config)
