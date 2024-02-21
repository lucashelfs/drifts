import os
import pandas as pd


from experiment import Experiment
from usp_stream_datasets import insects_datasets
from utils import fetch_original_dataframe, generate_experiment_filenames

from config import RESULTS_FOLDER

from visualizer import (
    fetch_experiment_change_points,
    plot_multiple_p_values,
    plot_original_data,
    fetch_top_n_accepted_attributes,
)

p_value = 0.05


# See for which datasets we ran the tests and collected some information
dataset_infos = []

for dataset in insects_datasets.keys():
    csv_file, metadata_file, _ = generate_experiment_filenames(
        dataset, results_folder=RESULTS_FOLDER
    )
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        dataset_infos.append({"dataset": dataset, "rows": df.shape[0]})


# this is the number of rows outputed by the tests
df_result_row_count = pd.DataFrame(dataset_infos)
df_result_row_count = df_result_row_count.sort_values("rows")


# these are the names of the datasets
tested_dataset_names = df_result_row_count.dataset.tolist()


#########################################
#########################################
############ VALIDATED AREA #############
#########################################
#########################################


# # VISUALIZING THE ORIGINAL DATA
# # LOOPING FOR A SINGLE SPECIES AND FOR SOME ATTRS
# dataset_name = tested_dataset_names[0]

# original_dataframe = fetch_original_dataframe(dataset_name)
# for species in original_dataframe["class"].unique().tolist()[:1]:
#     for attr in [f"Att{x}" for x in range(1, 2)]:
#         plot_original_data(dataset_name, species, attr)


dataset_name = "Incremental-gradual (bal.)"
attr = "Att27"

plot_original_data(dataset_name, attr=attr, species=False)


csv_file, _, _ = generate_experiment_filenames(
    dataset_name, results_folder=RESULTS_FOLDER
)
df_analysis = pd.read_csv(csv_file)
result_window_size = (df_analysis.end - df_analysis.start).max()

# Instance of experiment
exp = Experiment(dataset=dataset_name)
exp.prepare_insects_test()

# Check if nothing is missing...
original_df = fetch_original_dataframe(dataset_identifier=dataset_name)
assert exp.df_baseline.shape[0] + exp.df_stream.shape[0] == original_df.shape[0]


# Solving the mistery... Only one row was dropped
assert (
    len(df_analysis[df_analysis.attr == attr]) + exp.df_baseline.shape[0] + 1
    == exp.df_stream.shape[0]
)

# Properly posting the p-values
# plot_multiple_p_values(
#     dataset_name=dataset_name, p_value=0.05, top_n=5, index="start"
# )

plot_multiple_p_values(
    dataset_name=dataset_name,
    p_value=0.05,
    top_n=1,
)

######################################
######################################
############ TESTING AREA ############
######################################
######################################

# NICE!

# Plot original dfs and see if there are change points
# Plot original distribution vs p_value - for an attr of species
top_n = fetch_top_n_accepted_attributes(
    dataset_name=dataset_name, p_value=0.05, top_n=5
)

dengue = b"ae-aegypti-female"
for attr in top_n:
    plot_original_data(dataset_name, species=dengue, attr=attr)


# Does any baseline contain change points? NOPE!
for dataset_id in insects_datasets.keys():
    print(dataset_id, fetch_experiment_change_points(dataset_id))


#########################################
#########################################
############## TO BE REVISED ############
#########################################
#########################################
