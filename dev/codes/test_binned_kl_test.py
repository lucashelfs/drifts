import os
from experiment import Experiment


results_folder = os.getcwd() + "/results_of_5_median_binned_kl_experiments/"

# Sample test

dataset_name = "Incremental (bal.)"

exp = Experiment(
    dataset=dataset_name,
    batches=False,
    results_folder=results_folder,
    train_percentage=0.25,
    test_type="kl",
)

exp.prepare_insects_test()
exp.run_insects_test()
