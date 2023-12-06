import pandas as pd
import json
import time

from itertools import repeat
from scipy import stats
from multiprocessing import Pool
from usp_stream_datasets import load_insect_dataset, insects_datasets

from config import RESULTS_FOLDER


class Experiment:
    results_folder = RESULTS_FOLDER
    NUMBER_OF_POOLS = 8

    dataset_prefix = None
    metadata = {}

    def __init__(
        self,
        dataset: str,
        train_percentage: int = 0.25,
        DEBUG_SIZE=None,
        test_type="ks",
        stratified=False,
    ) -> None:
        self.dataset = dataset
        self.test_type = test_type
        self.train_percentage = train_percentage
        self.stratified = stratified
        self.dataset_prefix = self.results_folder + dataset.lower().replace(
            ".", ""
        ).replace("(", "").replace(")", "").replace(" ", "-")

        self.metadata["dataset"] = dataset
        self.metadata["execution_time"] = ({},)
        self.metadata["pools"] = self.NUMBER_OF_POOLS
        self.metadata["kind_of_test"] = self.test_type
        self.metadata["stratified"] = str(self.stratified)

        if DEBUG_SIZE:
            self.metadata["debug_size"] = DEBUG_SIZE
        else:
            self.DEBUG_SIZE = False

    def load_insects_dataframe(self):
        """Load dataframe from the insects datasets."""
        self.total_df = load_insect_dataset(
            insects_datasets[self.dataset]["filename"]
        )

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
        self.df_stream = self.total_df.loc[
            ~self.total_df.index.isin(baseline_index)
        ]
        self.df_stream = self.df_stream.rename_axis(
            "original_index"
        ).reset_index()

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
        self, start_idx, df_baseline, df_stream, attr, window_size, test="ks"
    ) -> dict:
        """Apply KS test on a window, to be used inside a multiprocessing pool.
        ref: https://stackoverflow.com/questions/63096168/how-to-apply-multiprocessing-to-a-sliding-window
        """
        start = start_idx + 1
        end = start + window_size
        baseline = df_baseline[attr]
        stream = df_stream[attr][start:end]
        if test == "ks":
            test_stat = stats.kstest(baseline, stream)
            return {
                "attr": attr,
                "start": start,
                "end": end,
                "original_start": df_stream["original_index"][start],
                "original_end": df_stream["original_index"][end],
                "p_value": test_stat.pvalue,
            }
        else:
            Exception("Undefined test!")

    def async_test(
        self,
        df_baseline: pd.DataFrame,
        df_stream: pd.DataFrame,
        attr: str,
        DEBUG_SIZE=None,
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
        STARTS = list(range(NUM_EL - WINDOW_SIZE - 1))
        if DEBUG_SIZE:
            STARTS = list(range(DEBUG_SIZE))

        with Pool(self.NUMBER_OF_POOLS) as p:
            result_multi = p.starmap(
                self.mp_window_test,
                zip(
                    STARTS,
                    repeat(df_baseline),
                    repeat(df_stream),
                    repeat(attr),
                    repeat(WINDOW_SIZE),
                ),
            )

        df_results = pd.DataFrame(result_multi)
        return df_results

    def set_attr_cols(self):
        """Set the atribute columns for the experiment."""
        self.attr_cols = [
            col for col in self.df_baseline.columns if col.startswith("Att")
        ]

    def async_test_for_multiple_attrs(self):
        """Test multiple attrs with a multithread approach."""
        results = []

        for attr in self.attr_cols:
            print(f"Testing for attr: {attr}")
            attr_start_time = time.time()
            attr_results = self.async_test(
                self.df_baseline,
                self.df_stream,
                attr,
                DEBUG_SIZE=self.DEBUG_SIZE,
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
        """Prepare insects dfs."""
        self.load_insects_dataframe()
        self.fetch_classes_and_minimal_class()
        self.set_window_size()
        self.create_baseline_dataframe()
        self.create_stream_dataframe()
        self.set_attr_cols()

    def run_insects_test(self):
        """Experiment logic run."""
        self.print_experiment_dfs()
        self.async_test_for_multiple_attrs()
