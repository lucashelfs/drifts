import os
import numpy as np

from experiment import Experiment
from kl import KLdivergence

# np.random.seed(42)

# dist1 = np.random.multivariate_normal(np.array([1, 1]), np.identity(2), 10000)
# dist2 = np.random.multivariate_normal(np.array([1, 1]), np.identity(2), 10000)

# KLdivergence(dist1, dist2)


# testar essa KL divergence com duas windows de experimento
results_folder = os.getcwd() + "/results_of_kl_experiments/"

# Sample test

dataset_name = "Incremental (bal.)"

exp = Experiment(
    dataset=dataset_name,
    batches=False,
    results_folder=results_folder,
    train_percentage=0.25,
)

exp.prepare_insects_test()

attr = "Att1"
start = 0
end = (
    start + exp.df_baseline.shape[0]
)  # incoerente com o outro teste que fazemos

baseline = exp.df_baseline[attr]
stream = exp.df_stream[attr][start:end]


KLdivergence(baseline, stream)


# from scipy.spatial import cKDTree as KDTree

# np.atleast_2d(baseline)
# KDTree(np.atleast_2d(baseline))
