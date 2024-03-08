import pandas as pd

from codes.dataset_experiment import DatasetExperiment
from codes.usp_stream_datasets import load_insect_dataset, insects_datasets


class InsectsExperiment(DatasetExperiment):
    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

        self.set_dataframes()
        self.set_test()

    def validate_given_attr(self):
        """Validate single attribute inputs."""
        if not self.attr.startswith("Att"):
            raise ("The given attribute doesn't match the insects DF pattern.")

    def load_insects_dataframe(self):
        """Load dataframe from the insects datasets."""
        self.total_df = load_insect_dataset(insects_datasets[self.dataset]["filename"])

        if self.attr:
            self.total_df = self.total_df[[self.attr, "class"]]

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

    def set_dataframes(self):
        """Abstract method for handling the baseline and stream dataframes."""
        self.load_insects_dataframe()
        self.fetch_classes_and_minimal_class()
        self.set_window_size()
        self.create_baseline_dataframe()
        self.create_stream_dataframe()

    def set_data_source(self):
        self.data_source = "insects"

    def set_results_dataset_filename(self):
        """Set the filename for the experiment results file."""
        self.dataset_prefix = self.results_folder + self.dataset.lower().replace(
            ".", ""
        ).replace("(", "").replace(")", "").replace(" ", "-")

    def fetch_classes_and_minimal_class(self):
        """Fetch classes available on the dataset and set the minimum size
        available.
        """
        self.classes = self.total_df["class"].unique().tolist()
        self.minimal_class = self.total_df["class"].value_counts().min()

    def set_window_size(self):
        """Set the size of the windows used on the experiment."""
        self.window_size = int(self.minimal_class * self.train_percentage)

    def set_test(self):
        """Set what else is needed for the test."""
        self.set_attr_cols()

    def set_attr_cols(self):
        """Set the atribute columns for the experiment."""
        if self.attr:
            self.desired_cols = [self.attr]
        else:
            self.desired_cols = [
                col for col in self.df_baseline.columns if col.startswith("Att")
            ]

    def print_experiment_info(self):
        """Print experiment information about insects specifically."""
        super().print_experiment_info()
        print("INSECTS INFORMATION")
        print(f"Minimal class: {self.minimal_class}")

    def run_test(self):
        """Logic for a run of the insect experiment."""
        self.print_experiment_info()
        self.async_test()
