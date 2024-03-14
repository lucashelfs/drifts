import pandas as pd
import os
import json
import time

from itertools import repeat
from scipy import stats
from multiprocessing import Pool
from usp_stream_datasets import load_insect_dataset, insects_datasets

from config import DEFAULT_RESULTS_FOLDER
from kl import calculate_kl_divergence_with_kde
from binning import (
    calculate_kl_with_dummy_bins,
    calculate_kl_with_median,
    calculate_js_with_dummy_bins,
    calculate_js_with_median,
)

from tqdm import tqdm

# The work below has an approach of data stream in individual new blocks and
# not in sequential windows.
# Wang, Zhixiong, and Wei Wang.
# "Concept drift detection based on Kolmogorov–Smirnov test.
# "Artificial Intelligence in China: Proceedings of the International
# Conference on Artificial Intelligence in China. Springer Singapore, 2020.

import multiprocessing.pool as mpp
from utils import istarmap


mpp.Pool.istarmap = istarmap


TEST_TYPES = ["ks", "kl_dummy", "kl_median", "js_dummy", "js_median"]


class Experiment:
    NUMBER_OF_POOLS = 8

    dataset_prefix = None
    metadata = {}

    def __init__(
        self,
        dataset: str,
        train_percentage: int = 0.25,
        DEBUG_SIZE=None,
        stratified=False,
        batches=False,
        results_folder=None,
        **kwargs,
    ) -> None:
        self.debug_size = DEBUG_SIZE

        # Handling better the results folder
        if not results_folder:
            self.results_folder = DEFAULT_RESULTS_FOLDER
        else:
            self.results_folder = results_folder

        self.set_result_directory()
        self.data_source = kwargs.get("data_source", False)
        if not self.data_source:
            raise (
                "The data source must be specified. Available: insects or synthetic."
            )

        # Fetch init data for the test type and its possible variables
        self.test_type = kwargs.get("test_type", False)
        if not self.test_type:
            raise ("Invalid test type!")

        # Check if test should be applied on one attribute
        self.attr = kwargs.get("attr", False)
        if self.attr:
            self.validate_given_attr()

        self.n_bins = kwargs.get("n_bins", False)  # validate the input with pydantic

        # TODO: fetch the median origin from kwargs

        if self.test_type not in TEST_TYPES:
            raise ("Invalid test!")

        # The other variables for the test are fetched here
        if (
            self.test_type.startswith("kl") or self.test_type.startswith("js")
        ) and not self.n_bins:
            raise ("Test is only available if n_bins is defined properly.")

        # Define specific configuration of the test
        self.batches = batches
        self.dataset = dataset
        self.train_percentage = train_percentage
        self.stratified = stratified

        # Set dataset prefix
        self.set_dataset_prefix()

        # Start the metadata structure
        self.write_initial_metadata()

    def set_dataset_prefix(self):
        """Set the dataset prefix for filename."""
        self.dataset_prefix = self.results_folder + self.dataset.lower().replace(
            ".", ""
        ).replace("(", "").replace(")", "").replace(" ", "-")

    def validate_given_attr(self):
        """Validate single attribute inputs."""
        if self.test_type == "insects":
            if not self.attr.startswith("Att"):
                raise ("The given attribute doesn't match the insects DF pattern.")

    def write_initial_metadata(self):
        """Write experiment metadata on the data structure."""
        self.metadata["dataset"] = self.dataset
        self.metadata["execution_time"] = ({},)
        self.metadata["pools"] = self.NUMBER_OF_POOLS
        self.metadata["kind_of_test"] = self.test_type
        self.metadata["stratified"] = str(self.stratified)
        self.metadata["batches"] = str(self.batches)
        if self.debug_size:
            self.metadata["debug_size"] = self.debug_size
        else:
            self.debug_size = False

    def set_result_directory(self):
        """Set the results directory if it does not exist."""
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print(f"Created directory: {self.results_folder}")
        else:
            print(f"The directory exists! {self.results_folder}")

    def load_insects_dataframe(self):
        """Load dataframe from the insects datasets."""
        self.total_df = load_insect_dataset(insects_datasets[self.dataset]["filename"])

        if self.attr:
            self.total_df = self.total_df[[self.attr, "class"]]

    def fetch_classes_and_minimal_class(self):
        """Fetch classes available on the dataset and set the minimum size
        available.
        """
        self.classes = self.total_df["class"].unique().tolist()
        self.minimal_class = self.total_df["class"].value_counts().min()

    def set_window_size(self):
        """Set the size of the windows used on the experiment."""
        self.window_size = int(self.minimal_class * self.train_percentage)

    def create_baseline_dataframe(self):
        """Create a baseline dataframe for the experiment."""
        if not self.stratified:
            baseline_dfs = [
                self.total_df[self.total_df["class"] == species].iloc[
                    : self.window_size,
                ]
                for species in self.classes
            ]
            self.df_baseline = pd.concat(baseline_dfs)
            self.df_baseline = self.df_baseline.rename_axis(
                "original_index"
            ).reset_index()
        else:
            Exception("Stratified baseline not implemented.")

    def create_stream_dataframe(self):
        """Create a stream dataframe for the experiment."""
        baseline_index = self.df_baseline.index.tolist()
        self.df_stream = self.total_df.loc[~self.total_df.index.isin(baseline_index)]
        self.df_stream = self.df_stream.rename_axis("original_index").reset_index()

    def print_experiment_dfs(self):
        print(f"DF Total: {self.total_df.shape}")
        print(f"DF baseline: {self.df_baseline.shape}")
        print(f"DF stream: {self.df_stream.shape}")
        print(f"Minimal class: {self.minimal_class}")

    def write_metadata(self):
        """Write the dict with experiment metadata on a json file."""
        json_object = json.dumps(self.metadata, indent=4)
        with open(f"{self.dataset_prefix}_metadata.json", "w") as outfile:
            outfile.write(json_object)
        return

    def mp_window_test(
        self,
        start_idx: int,
        df_baseline: pd.DataFrame,
        df_stream: pd.DataFrame,
        attr: str,
        window_size: int,
    ) -> dict:
        """Apply test on a window, to be used inside a multiprocessing pool.
        ref: https://stackoverflow.com/questions/63096168/how-to-apply-multiprocessing-to-a-sliding-window
        """
        start = start_idx + 1
        end = start + window_size
        baseline = df_baseline[attr]
        stream = df_stream[attr][start:end]

        if self.test_type == "ks":
            test_stat = stats.kstest(baseline, stream)
            return {
                "attr": attr,
                "start": start,
                "end": end,
                "original_start": df_stream["original_index"][start],
                "original_end": df_stream["original_index"][end],
                "p_value": test_stat.pvalue,
            }
        elif self.test_type == "kl_median":
            distance = calculate_kl_with_median(
                baseline, stream, median_origin="both", n_bins=self.n_bins
            )
            return {
                "attr": attr,
                "start": start,
                "end": end,
                "original_start": df_stream["original_index"][start],
                "original_end": df_stream["original_index"][end],
                "distance": distance,
            }
        elif self.test_type == "kl_dummy":
            distance = calculate_kl_with_dummy_bins(
                baseline, stream, n_bins=self.n_bins
            )
            return {
                "attr": attr,
                "start": start,
                "end": end,
                "original_start": df_stream["original_index"][start],
                "original_end": df_stream["original_index"][end],
                "distance": distance,
            }
        elif self.test_type == "js_median":
            distance = calculate_js_with_median(
                baseline, stream, median_origin="both", n_bins=self.n_bins
            )
            return {
                "attr": attr,
                "start": start,
                "end": end,
                "original_start": df_stream["original_index"][start],
                "original_end": df_stream["original_index"][end],
                "distance": distance,
            }
        elif self.test_type == "js_dummy":
            distance = calculate_js_with_dummy_bins(
                baseline, stream, n_bins=self.n_bins
            )
            return {
                "attr": attr,
                "start": start,
                "end": end,
                "original_start": df_stream["original_index"][start],
                "original_end": df_stream["original_index"][end],
                "distance": distance,
            }
        else:
            Exception("Undefined test!")

    def async_test(
        self,
        df_baseline: pd.DataFrame,
        df_stream: pd.DataFrame,
        attr: str,
        batches=False,
        debug_size=None,
    ) -> pd.DataFrame:
        """Create a pool and execute the test on a range of windows
        on the stream dataframe.

        :param df_baseline: Baseline dataframe.
        :type df_baseline: pd.DataFrame
        :param df_stream: Stream dataframe.
        :type df_stream: pd.DataFrame
        :param attr: Attribute colum to be tested.
        :type attr: str
        :param DEBUG_SIZE: Size to be tested in debug, defaults to None
        :type DEBUG_SIZE: int, optional
        :return: Dataframe with results of the applied tests.
        :rtype: pd.DataFrame
        """
        NUM_EL = len(df_stream[attr])
        WINDOW_SIZE = len(df_baseline[attr])

        if not batches:
            STARTS = list(range(NUM_EL - WINDOW_SIZE - 1))
        else:
            STARTS = list(range(0, NUM_EL - WINDOW_SIZE - 1, WINDOW_SIZE))

        if debug_size:
            if not batches:
                STARTS = list(range(debug_size))
            else:
                STARTS = list(range(0, debug_size, WINDOW_SIZE))

        with Pool(self.NUMBER_OF_POOLS) as p:
            result_multi = p.starmap(
                self.mp_window_test,
                tqdm(
                    zip(
                        STARTS,
                        repeat(df_baseline),
                        repeat(df_stream),
                        repeat(attr),
                        repeat(WINDOW_SIZE),
                    ),
                    total=len(STARTS),
                ),
            )

        df_results = pd.DataFrame(result_multi)
        return df_results

    def set_attr_cols(self):
        """Set the atribute columns for the experiment."""
        if self.attr:
            self.desired_cols = [self.attr]
        else:
            self.desired_cols = [
                col for col in self.df_baseline.columns if col.startswith("Att")
            ]

    def async_test_for_desired_attrs(self):
        """Test for the desired attrs with a multithread approach."""
        results = []

        for attr in self.desired_cols:
            print(f"Testing for attr: {attr}")
            attr_start_time = time.time()
            attr_results = self.async_test(
                self.df_baseline,
                self.df_stream,
                attr,
                debug_size=self.debug_size,
                batches=self.batches,
            )
            attr_end_time = time.time()
            elapsed_attr_time = attr_end_time - attr_start_time
            self.metadata["execution_time"][0][attr] = elapsed_attr_time
            results.append(attr_results)

        self.metadata["Class size"] = self.window_size
        self.metadata["Baseline size"] = self.df_baseline.shape[0]
        self.metadata["Stream size"] = self.df_stream.shape[0]
        self.write_metadata()

        dataset_results = pd.concat(results)
        dataset_results.to_csv(self.dataset_prefix + ".csv", index=None)

    def prepare_insects_test(self):
        """Prepare insects dataframes."""
        if self.data_source == "insects":
            self.load_insects_dataframe()
            self.fetch_classes_and_minimal_class()
            self.set_window_size()
            self.create_baseline_dataframe()
            self.create_stream_dataframe()
            self.set_attr_cols()
        else:
            raise ("Method not available for the given data source.")

    def run_insects_test(self):
        """Logic for a run of the insect experiment."""
        if self.data_source == "insects":
            self.print_experiment_dfs()
            self.async_test_for_desired_attrs()
        else:
            raise ("Method not available for the given data source.")
