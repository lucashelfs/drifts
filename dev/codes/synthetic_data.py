import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Generate data for 3 distributions
# random_state = np.random.RandomState(seed=42)
# dist_a = random_state.normal(0.8, 0.05, 1000)
# dist_b = random_state.normal(0.4, 0.02, 1000)
# dist_c = random_state.normal(0.6, 0.1, 1000)

# dist_a = random_state.normal(0.8, 0.05, 1500)
# dist_b = random_state.normal(0.7, 0.02, 1000)
# dist_c = random_state.normal(0.6, 0.1, 1000)

# Concatenate data to simulate a data stream with 2 drifts
# stream = np.concatenate((dist_a, dist_b, dist_c))


# # Auxiliary function to plot and verify the data
# def plot_data(dist_a, dist_b, dist_c, drifts=None):
#     fig = plt.figure(figsize=(7, 3), tight_layout=True)
#     gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
#     ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
#     ax1.grid()
#     ax1.plot(stream, label="Stream")
#     ax2.grid(axis="y")
#     ax2.hist(dist_a, label=r"$dist_a$")
#     ax2.hist(dist_b, label=r"$dist_b$")
#     ax2.hist(dist_c, label=r"$dist_c$")
#     if drifts is not None:
#         for drift_detected in drifts:
#             ax1.axvline(drift_detected, color="red")
#     plt.show()

## Check Table 3 with the most common Synthetic Datasets from
# Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019).
# Learning under Concept Drift: A Review. In IEEE Transactions on Knowledge and Data Engineering (Vol. 31, Issue 12, pp. 2346–2363).
# IEEE Computer Society. https://doi.org/10.1109/TKDE.2018.2876857


class SyntheticData:
    def __init__(self, **kwargs) -> None:
        self.distributions = kwargs.get("distributions", [])
        self.drifts = kwargs.get("drifts", None)
        self.create_stream()
        self.set_change_points()

    def create_stream(self):
        self.stream = np.concatenate(tuple(self.distributions))

    def set_change_points(self):
        self.change_points = []
        change_point = 0
        for i in range(len(self.distributions) - 1):
            distribution = self.distributions[i]
            change_point += len(distribution)
            self.change_points.append(change_point)

    def get_stream(self):
        return self.stream

    def plot_stream(self):
        fig = plt.figure(figsize=(7, 3), tight_layout=True)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
        ax1.grid()

        ax1.plot(self.stream, label="Stream")
        ax2.grid(axis="y")

        for i, dist in enumerate(self.distributions):
            ax2.hist(dist, label=f"dist_{i+1}")

        if self.drifts is not None:
            for drift_detected in self.drifts:
                ax1.axvline(drift_detected, color="red")

        ax2.legend()
        plt.show()


# if __name__ == "__main__":
# create_stream()
# plot_data(dist_a, dist_b, dist_c)
