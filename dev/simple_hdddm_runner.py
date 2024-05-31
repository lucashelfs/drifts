"""
This plots the values for the HDDDM from the Menelaus implementation, plotting the difference between
epsilon and the threshold.
"""

from dev.config import insects_datasets, load_insect_dataset
import numpy as np
import pandas as pd

from menelaus.data_drift.hdddm import HDDDM
from dev.common import load_and_prepare_dataset

np.random.seed(1)


from dev.ddm import fetch_hdddm_drifts


if __name__ == "__main__":
    print(fetch_hdddm_drifts())
