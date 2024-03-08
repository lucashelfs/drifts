import os
from codes.insects_experiment import InsectsExperiment

n_bins = 5
dataset_name = "Incremental (bal.)"
attr = "Att31"
test_type = "kl_dummy"
binning_type = "median"
median_origin = "both"

results_folder = os.getcwd() + f"/TEST-results_of_{test_type}_{n_bins}_{attr}/"
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


ie = InsectsExperiment(**config)
ie.run_test()


# # Generate GIF of stream binning
# # create_binning_timeline(exp.df_baseline, exp.df_stream, **config)

# # # Plot values for that attribute
# # plot_kl_values(**config)

# # TODO: improve below
# # Plot first baseline bin
# # WINDOW_SIZE = len(exp.df_baseline[attr])
# # plot_binning(exp.df_baseline[attr],
# #              exp.df_stream[attr][:WINDOW_SIZE],
# #              **config)
