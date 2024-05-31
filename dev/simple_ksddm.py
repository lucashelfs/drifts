"""
Simple KSDDM implementation.
"""

import copy
import pandas as pd
from scipy import stats


class SimpleKSDDM:
    def __init__(self, mean_threshold=0.05) -> None:
        self.drift_state = []
        self.all_p_values = []
        self.current_batch = None
        self.batches_since_reset = 0
        self.batches = 0
        self.p_value_diff = 0
        self.mean_threshold = mean_threshold
        self.reference = None
        self.p_values = pd.Series()
        self.p_values_diff = []
        self.last_p_value_series = None

    def set_reference(self, reference):
        self.reference = copy.deepcopy(reference)
        self.batches += 1

    def calculate_ks_distance(
        self,
    ):
        feature_p_values = []

        for feature in self.reference.columns:
            test_stat = stats.ks_2samp(
                self.reference[feature], self.current_batch[feature]
            ).pvalue
            feature_p_values.append(test_stat)

        self.p_values = pd.Series(feature_p_values)
        self.all_p_values.append(self.p_values)

        if self.last_p_value_series is not None:
            self.p_value_diff = abs(self.p_values - self.last_p_value_series)
            self.p_values_diff.append(self.p_value_diff)
            self.last_p_value_series = self.p_values
        else:
            self.p_value_diff = abs(self.p_values)
            self.last_p_value_series = self.p_values

    def update(self, batch):
        self.batches += 1
        self.batches_since_reset += 1
        self.current_batch = batch

        if self.batches_since_reset >= 1:
            self.calculate_ks_distance()
            self.determine_drift()

        else:
            self.drift_state.append("")

    def determine_drift(self):
        if self.p_values.mean() < self.mean_threshold:
            self.reference = self.current_batch
            self.batches_since_reset = 0
            self.drift_state.append("drift")
        else:
            self.reference = pd.concat([self.reference, self.current_batch])
            self.drift_state.append(None)
