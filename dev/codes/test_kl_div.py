import os
import numpy as np

from experiment import Experiment
from kl import calculate_kl_divergence_with_kde, KLdivergence
from new_kl import scipy_estimator

# np.random.seed(42)

# dist1 = np.random.multivariate_normal(np.array([1, 1]), np.identity(2), 10000)
# dist2 = np.random.multivariate_normal(np.array([1, 1]), np.identity(2), 10000)


# testar essa KL divergence com duas windows de experimento
results_folder = os.getcwd() + "/results_of_kl_experiments/"

# Sample test

# dataset_name = "Incremental (bal.)" # this one leads to zero div
dataset_name = "Abrupt (bal.)"

exp = Experiment(
    dataset=dataset_name,
    batches=False,
    results_folder=results_folder,
    train_percentage=0.25,
)

exp.prepare_insects_test()


## Teste com stream e baseline diferentes
attr = "Att1"
start = 0
end = start + exp.df_baseline.shape[0]  # tem essa restrição aqui do tamanho

baseline = exp.df_baseline[attr]
stream = exp.df_stream[attr][start:end]

## DJ KHALED: https://stackoverflow.com/questions/71218637/how-to-calculate-a-probability-distribution-from-an-array-with-scipy
import scipy
from scipy.stats import rv_histogram

# Make distribution objects of the histograms
histogram_dist_train = rv_histogram(np.histogram(baseline, bins="auto"))
histogram_dist_valid = rv_histogram(np.histogram(stream, bins="auto"))

# Generate arrays of pdf evaluations
X1 = np.linspace(np.min(baseline), np.max(baseline), 1000)
X2 = np.linspace(np.min(stream), np.max(stream), 1000)
rvs_train = [histogram_dist_train.pdf(x) for x in X1]
rvs_valid = [histogram_dist_valid.pdf(x) for x in X2]


# Calculate the Kullback–Leibler divergence between the different datasets
entropy_train_valid = scipy.special.rel_entr(rvs_train, rvs_valid)
kl_div_train_valid = np.sum(entropy_train_valid)

# Print the values of the Kullback–Leibler divergence
print(
    f"Kullback–Leibler divergence between training and validation dataset: {kl_div_train_valid}"
)


############ KL TESTS

# With KDE

kl1 = calculate_kl_divergence_with_kde(baseline, stream)
# kl2 = calculate_kl_divergence_with_kde(baseline, stream, bigger=True)


# Without KDE, needs reshaping

s1 = baseline.values.reshape(-1, 1)
s2 = stream.values.reshape(-1, 1)

# scipy_estimator(s1, s2)
KLdivergence(s1, s2)


# s1 = baseline.values.reshape(1, -1)
# s2 = stream.values.reshape(1, -1)
# KLdivergence(s1, s2)

# ## Teste com o mesmo tamanho original do teste do experimento
# attr = "Att1"
# start = 0
# end = start + exp.window_size

# baseline = exp.df_baseline[attr]
# stream = exp.df_stream[attr][start:end]

# kl2 = calculate_kl_divergence_with_kde(baseline, stream)


##################################################################
###### Testing more about the problem with division by zero ######
##################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Generate data for 3 distributions
random_state = np.random.RandomState(seed=42)
dist_a = random_state.normal(0.8, 0.05, 1000)
dist_b = random_state.normal(0.4, 0.02, 1000)
dist_c = random_state.normal(0.6, 0.1, 1000)

# Testing the estimators
scipy_estimator(dist_a.reshape(-1, 1), dist_b.reshape(-1, 1))
KLdivergence(dist_a.reshape(-1, 1), dist_b.reshape(-1, 1))


##################################################################
###### Creating the synthetic with the distributions stream ######
##################################################################

# Concatenate data to simulate a data stream with 2 drifts
stream = np.concatenate((dist_a, dist_b, dist_c))


# Auxiliary function to plot and verify the data
def plot_data(dist_a, dist_b, dist_c, drifts=None):
    fig = plt.figure(figsize=(7, 3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.grid()
    ax1.plot(stream, label="Stream")
    ax2.grid(axis="y")
    ax2.hist(dist_a, label=r"$dist_a$")
    ax2.hist(dist_b, label=r"$dist_b$")
    ax2.hist(dist_c, label=r"$dist_c$")
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color="red")
    plt.show()


plot_data(dist_a, dist_b, dist_c)
