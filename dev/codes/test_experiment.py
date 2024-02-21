import os
from experiment import Experiment


results_folder = os.getcwd() + "/results_of_basic_experiment/"

# Sample test

dataset_name = "Incremental (bal.)"

exp = Experiment(
    dataset=dataset_name,
    batches=False,
    results_folder=results_folder,
    train_percentage=0.25,
)

exp.prepare_insects_test()
exp.run_insects_test()


# # Visualizing...
# from visualizer import plot_multiple_p_values


# # attr = "Att27"
# # plot_original_data(dataset_name, attr=attr, species=False)


# plot_multiple_p_values(
#     dataset_name=dataset_name,
#     p_value=0.05,
#     top_n=5,
#     results_folder=results_folder,
# )
