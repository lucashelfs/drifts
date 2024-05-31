"""
This plots the values for the jsddm from the Menelaus implementation, plotting the difference between
epsilon and the threshold.
"""


import numpy as np
import pandas as pd

# from dev.ddm import fetch_jsddm_drifts
from dev.JSDDM import JSDDM
from dev.common import load_and_prepare_dataset

np.random.seed(1)




# if __name__ == "__main__":
#     print(fetch_jsddm_drifts())
