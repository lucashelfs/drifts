import numpy as np
import scipy, time
import warnings
import matplotlib.pyplot as plt
from random import random
from collections import Counter
from scipy.interpolate import interp1d
from scipy import stats, integrate

warnings.filterwarnings("ignore")


# returns the step function value at each increment of the CDF
def cumcount_reduced(arr):
    """Cumulative count of a array"""
    sorted_arr = np.array(sorted(arr))
    counts = np.zeros(len(arr))
    rolling_count = 0
    for idx, elem in enumerate(sorted_arr):
        rolling_count += 1
        counts[idx] = rolling_count
    counts /= len(counts)
    counts -= 1 / (2 * len(counts))
    return (sorted_arr, counts)


# takes two datasets to estimate the relative entropy between their PDFs
# we use eps=10^-11, but it could be defined as < the minimal interval between data points
def KLD_PerezCruz(P, Q, eps=1e-11):
    """KL divergence calculation"""
    P = sorted(P)
    Q = sorted(Q)
    P_positions, P_counts = cumcount_reduced(P)
    Q_positions, Q_counts = cumcount_reduced(Q)
    x_0 = np.min([P_positions[0], Q_positions[0]]) - 1
    P_positions = np.insert(P_positions, 0, [x_0])
    P_counts = np.insert(P_counts, 0, [0])
    Q_positions = np.insert(Q_positions, 0, [x_0])
    Q_counts = np.insert(Q_counts, 0, [0])
    x_np1 = np.max([P_positions[-1], Q_positions[-1]]) + 1
    P_positions = np.append(P_positions, [x_np1])
    P_counts = np.append(P_counts, [1])
    Q_positions = np.append(Q_positions, [x_np1])
    Q_counts = np.append(Q_counts, [1])
    f_P = interp1d(P_positions, P_counts)
    f_Q = interp1d(Q_positions, Q_counts)
    X = P_positions[1:-2]
    values = (f_P(X) - f_P(X - eps)) / (f_Q(X) - f_Q(X - eps))
    filt = (
        (values != 0.0) & ~(np.isinf(values)) & ~(np.isnan(values))
    )  # works because of this here...
    values_filter = values[filt]
    out = (np.sum(np.log(values_filter)) / len(values_filter)) - 1.0
    return out


###################

import os
import numpy as np

from experiment import Experiment

results_folder = os.getcwd() + "/results_of_kl_experiments/"
dataset_name = "Abrupt (bal.)"

exp = Experiment(
    dataset=dataset_name,
    batches=False,
    results_folder=results_folder,
    train_percentage=0.25,
)

exp.prepare_insects_test()


attr = "Att1"
start = 0
end = start + exp.df_baseline.shape[0]  # tem essa restrição aqui do tamanho

baseline = exp.df_baseline[attr]
stream = exp.df_stream[attr][start:end]


################################


data = np.genfromtxt("testdata.txt")
time = [x[0] for x in data]
proc1 = [x[1] for x in data]
proc2 = [x[2] for x in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

ax1.set(aspect=10)
ax1.set_title("process 1")
ax1.set_xlabel("time (s)")
ax1.plot(time, proc1)

ax2.set(aspect=10)
ax2.set_title("process 2")
ax2.set_xlabel("time (s)")
ax2.plot(time, proc1)

plt.show()


##########

time = baseline.index.tolist()
proc1 = baseline.values
proc2 = stream.values


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

ax1.set(aspect=10)
ax1.set_title("process 1")
ax1.set_xlabel("time (s)")
ax1.plot(time, proc1)

ax2.set(aspect=10)
ax2.set_title("process 2")
ax2.set_xlabel("time (s)")
ax2.plot(time, proc1)

plt.show()


hist1 = plt.hist(proc1, 100, alpha=0.5, label="1", density=True)
hist2 = plt.hist(proc2, hist1[1], alpha=0.5, label="2", density=True)

# # Viz
# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.show()

kld_value = KLD_PerezCruz(proc1, proc2)
print(f"KLD = {kld_value:.4f}")

# dt = (time[1] - time[0])
# kld_rate = kld_value/dt
# print(f"KLD per unit of time = {kld_rate:.2f} s^-1")


bin_size = hist1[1][1] - hist1[1][0]
p = hist1[0] * bin_size
q = hist2[0] * bin_size

kld_std = np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))

print(f"standard KLD = {kld_std:.4f}")
