import json
import os
import time
import pandas as pd

from abc import ABC, abstractmethod
from itertools import repeat
from scipy import stats
from multiprocessing import Pool

from codes.binning import (
    calculate_kl_with_dummy_bins,
    calculate_kl_with_median,
    calculate_js_with_dummy_bins,
    calculate_js_with_median,
)

from tqdm import tqdm

# TODO: make the test types as an ENUM
TEST_TYPES = ["ks", "kl_dummy", "kl_median", "js_dummy", "js_median"]


class BaseExperiment(ABC):
    """
    Base experiment class, with a multithreaded approach for testing drifts between a baseline dataframe and a stream one.

    References
    ----------

    Wang, Zhixiong, and Wei Wang. "Concept drift detection based on Kolmogorov–Smirnov test."
    Artificial Intelligence in China: Proceedings of the International Conference on Artificial Intelligence in China.
    Springer Singapore, 2020.

    """

    NUMBER_OF_POOLS = 8

    dataset_prefix = None
    metadata = {}

    def __init__(
        self,
        **kwargs,
    ) -> None:

        # Fetch init data for the test type and its possible variables
        self.test_type = kwargs.get("test_type", False)
        if not self.test_type:
            raise ("Invalid test type!")

        # Check if test should be applied on only one attribute
        # TODO: validate if this can be a DatasetExperiment method
        self.attr = kwargs.get("attr", False)
        if self.attr:
            self.validate_given_attr()

        # TODO: validate the input with pydantic
        self.n_bins = kwargs.get("n_bins", False)

        # TODO: fetch the median origin from kwargs
        if self.test_type not in TEST_TYPES:
            raise ("Invalid test!")

        # The other variables for the test are fetched here
        if (
            self.test_type.startswith("kl") or self.test_type.startswith("js")
        ) and not self.n_bins:
            raise ("Test is only available if n_bins is defined properly.")

        # Define specific configuration of the test
        self.batches = kwargs.get("batches", False)
        self.stratified = kwargs.get("stratified", False)
        self.train_percentage = kwargs.get("train_percentage", 0.25)
        self.results_folder = kwargs.get("results_folder", False)
        self.debug_size = kwargs.get("debug_size", None)

        # Set some configs
        self.set_data_source()
        self.set_result_directory()
        self.write_metadata(initial=True)
        self.set_results_dataset_filename()

    def write_metadata(self, initial: bool = False):
        """Write experiment metadata on its data structure."""
        if initial:
            self.metadata["execution_time"] = ({},)
            self.metadata["pools"] = self.NUMBER_OF_POOLS
            self.metadata["kind_of_test"] = self.test_type
            self.metadata["stratified"] = str(self.stratified)
            self.metadata["batches"] = str(self.batches)
            self.metadata["data_source"] = str(self.data_source)
            if self.debug_size:
                self.metadata["debug_size"] = self.debug_size
            else:
                self.debug_size = False
        else:
            json_object = json.dumps(self.metadata, indent=4)
            with open(f"{self.dataset_prefix}_metadata.json", "w") as outfile:
                outfile.write(json_object)
            return

    def set_result_directory(self):
        """Set the results directory if it does not exist."""
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print(f"Created directory: {self.results_folder}")
        else:
            print(f"The directory exists! {self.results_folder}")

    def print_experiment_info(self):
        """Print experiment information about the main dataframes used on the tests."""
        print("GENERAL INFORMATION")
        print(f"DF baseline: {self.df_baseline.shape}")
        print(f"DF stream: {self.df_stream.shape}")

    def mp_window_test(
        self,
        start_idx: int,
        df_baseline: pd.DataFrame,
        df_stream: pd.DataFrame,
        attr: str,
        window_size: int,
    ) -> dict:
        """Apply test on a window, to be used inside a multiprocessing pool.

        Args:
            start_idx (int): The starting index of the window on the stream dataframe
            df_baseline (pd.DataFrame): Baseline dataframe
            df_stream (pd.DataFrame): Stream dataframe
            attr (str): The attribute column on the dataframe being tested
            window_size (int): The testing window size

        Returns:
            dict: Dictionary with the test params and results

        References:

        - https://stackoverflow.com/questions/63096168/how-to-apply-multiprocessing-to-a-sliding-window
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

    def async_test_runner(
        self,
        df_baseline: pd.DataFrame,
        df_stream: pd.DataFrame,
        attr: str,
        batches=False,
        debug_size=None,
    ) -> pd.DataFrame:
        """Create a pool and execute the test on a range of windows
        on the stream dataframe.

        Args:
            df_baseline (pd.DataFrame): Baseline dataframe.
            df_stream (pd.DataFrame): Stream dataframe.
            attr (str): Attribute column to be tested.
            batches (bool, optional): If the experiment uses sliding windows or batches. Defaults to False.
            debug_size (_type_, optional): A debug size for testing. Defaults to None.

        Returns:
            pd.DataFrame: _description_
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

    def async_test(self):
        """Test for the desired attrs with a multithread approach."""
        results = []

        for attr in self.desired_cols:
            print(f"Testing for attr: {attr}")
            attr_start_time = time.time()
            attr_results = self.async_test_runner(
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

    @abstractmethod
    def run_test(self):
        """Logic for a experiment run."""
        raise NotImplementedError("This method has not been implemented.")

    @abstractmethod
    def set_dataframes(self):
        """Abstract method for handling the baseline and stream dataframes."""
        raise NotImplementedError("This method has not been implemented.")

    @abstractmethod
    def set_data_source(self):
        raise NotImplementedError("This method has not been implemented.")

    @abstractmethod
    def set_results_dataset_filename(self):
        """Set the filename for the experiment results file."""
        raise NotImplementedError("This method has not been implemented.")
