import os
from codes.synthetic_experiment import SyntheticExperiment, SynthethicVisualizer

import numpy as np


# Synthetic data params
random_state = np.random.RandomState(seed=42)
visualize_stream = True
synthetic_params = [
    random_state.normal(0.8, 0.05, 2000),
    random_state.normal(0.7, 0.05, 2000),
    random_state.normal(0.6, 0.1, 2000),
]

window_size = 1000

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

# JS/KL config
test_type = "js_dummy"
n_bins = 5
results_folder = os.getcwd() + f"/TEST-SYNTHETIC-results_of_{test_type}/"

config = dict(
    batches=False,
    results_folder=results_folder,
    test_type=test_type,
    window_size=window_size,
    n_bins=n_bins,
    visualize_stream=visualize_stream,
    distributions=synthetic_params,
)

if __name__ == "__main__":
    # Set the experiment
    se = SyntheticExperiment(**config)
    se.run_test()

    # Set a visualizer for the experiment result
    sv = SynthethicVisualizer(se)

    # Plot the obtained results
    sv.plot_result_values(**config)
