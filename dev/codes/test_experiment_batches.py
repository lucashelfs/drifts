import os
from experiment import Experiment


results_folder = os.getcwd() + "/results_of_testing_batches/"

# For BIG datasets, batches are a nice thing - is fast as we can see below
# but we need to add a check on experiment if the smaller classes are too small
# this has an implication on the performance of the test, because of the window size

dataset_name = "Incremental (imbal.)"

# exp = Experiment(
#     dataset=dataset_name,
#     batches=True,
#     results_folder=results_folder,
#     train_percentage=0.25,
# )

# exp.prepare_insects_test()
# exp.run_insects_test()


# Visualizing...
from visualizer import plot_multiple_p_values


# attr = "Att27"
# plot_original_data(dataset_name, attr=attr, species=False)


plot_multiple_p_values(
    dataset_name=dataset_name,
    p_value=0.05,
    top_n=5,
    results_folder=results_folder,
)


## Other test but on a dataset with abrupt changes
dataset_name = "Abrupt (imbal.)"

# exp = Experiment(
#     dataset=dataset_name,
#     batches=True,
#     results_folder=results_folder,
#     train_percentage=0.25,
# )

# exp.prepare_insects_test()
# exp.run_insects_test()


# The interesting fact here is that for the abrupt dataset changes
# the visualizing works properly because some attrs were accepted

from visualizer import plot_multiple_p_values

plot_multiple_p_values(
    dataset_name=dataset_name,
    p_value=0.05,
    top_n=5,
    results_folder=results_folder,
)


# We need to add a check on experiment if the smaller classes are too small
# this has an implication on the performance of the test, because of the window size
# example below

dataset_name = "Out-of-control"

# exp = Experiment(
#     dataset=dataset_name,
#     batches=True,
#     results_folder=results_folder,
#     train_percentage=0.25,
# )

# exp.prepare_insects_test()
# exp.run_insects_test()
