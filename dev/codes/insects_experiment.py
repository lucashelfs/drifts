import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from codes.dataset_experiment import DatasetExperiment
from codes.usp_stream_datasets import load_insect_dataset, insects_datasets


class InsectsExperiment(DatasetExperiment):
    """Insects experiments abstraction."""

    def __init__(
        self,
        **kwargs,
    ) -> None:

        self.set_dataset(**kwargs)

        super().__init__(
            **kwargs,
        )

        self.set_dataframes()
        self.set_test()

    def set_dataset(self, **kwargs):
        self.dataset = kwargs.get("dataset", None)
        if not self.dataset:
            raise Exception("The dataset mus be speficied for dataset experiments!")

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


class InsectsVisualizer:
    """Visualizer for the results of a insect experiment."""

    sns.set(rc={"figure.figsize": (12, 6)})

    def __init__(self, experiment) -> None:
        assert isinstance(experiment, InsectsExperiment)
        self.experiment = experiment

    def fetch_experiment_filename(self):
        """Create the output filenames from the given experiment."""
        results_file = self.experiment.dataset_prefix + ".csv"
        metadata_file = self.experiment.dataset_prefix + "metadata.json"
        plot_file = self.experiment.dataset_prefix + ".jpg"
        return results_file, metadata_file, plot_file

    def plot_result_values(self, **kwargs):
        """Plot the results for the given experiment."""

        index = "end"
        savefig = kwargs.get("savefig", False)
        attr = kwargs.get("attr", False)

        if attr:
            attr_list = [attr]

        if not attr:
            print(
                "Attribute not defined for result values. Trying to fetch from experiment."
            )
            attr_list = self.experiment.desired_cols

        csv_file, _, plot_file = self.fetch_experiment_filename()
        df_analysis = pd.read_csv(csv_file)

        test_type = self.experiment.test_type

        if test_type == "ks":
            result_col = "p_value"
        else:
            result_col = "distance"

        df_plot = df_analysis[(df_analysis["attr"].isin(attr_list))][
            [result_col, "attr", "start", "end"]
        ]

        dataset = kwargs.get("dataset", False)

        if not dataset:
            raise Exception("Missing dataset on experiment visualizer!")

        if df_plot.empty:
            print(
                "The dataframe for plotting is empty! The full analysis will be plotted."
            )
            df_plot = df_analysis

        n_bins = kwargs.get("n_bins", False)

        if not n_bins:
            title = f"Distance for {dataset} dataset indexed by {index} - {attr} - Test type: {test_type}"
        else:
            title = f"Distance for {dataset} dataset indexed by {index} - {attr} - bins: {n_bins} - Test type: {test_type}"

        plt.clf()
        plt.title(title)
        g = sns.lineplot(x=index, y=result_col, hue="attr", data=df_plot)
        change_points = self.fetch_change_points()
        stream_change_points = change_points["stream"]
        if stream_change_points:
            for change_point in stream_change_points:
                g.axvline(change_point["reindexed"], ls="--", c="yellow")
        if savefig:
            plt.savefig(plot_file)
        plt.show()

    def fetch_change_points(self) -> list:
        """Fetch reindexed change points on baseline and stream datasets."""
        change_points = insects_datasets[self.experiment.dataset].get(
            "change_point", []
        )
        cps = {"baseline": [], "stream": []}

        # Fetch for stream
        for change_point in change_points:
            stream_change_point = self.experiment.df_stream.original_index[
                self.experiment.df_stream.original_index == change_point
            ].index.tolist()
            if len(stream_change_point) > 0:
                cps["stream"].append(
                    {"original": change_point, "reindexed": stream_change_point[0]}
                )
        # Fetch for baseline
        for change_point in change_points:
            baseline_change_point = self.experiment.df_baseline.original_index[
                self.experiment.df_baseline.original_index == change_point
            ].index.tolist()
            if len(baseline_change_point) > 0:
                cps["baseline"].append(
                    {
                        "original": change_point,
                        "reindexed": baseline_change_point[0],
                    }
                )
        return cps

    def plot_original_data(self, **kwargs):
        """Create the lineplot of an attribute of a species on the dataset of the experiment."""

        original_dataframe = self.experiment.total_df
        dataset_name = self.experiment.dataset
        change_points = self.fetch_change_points()

        species = kwargs.get("species", False)
        attr = kwargs.get("attr", False)

        if not attr:
            print("Please set the attribute to be displayed.")
            return

        if species:
            title = (
                f"Original distribution - {species} - {dataset_name} dataset - {attr}"
            )
            df = original_dataframe[original_dataframe["class"] == species].iloc[:,]
        else:
            title = (
                f"Original distribution - all species - {dataset_name} dataset - {attr}"
            )
            df = original_dataframe

        if attr not in df.columns.tolist():
            print("The attribute is not present on the Dataframe.")
            print(f"The available Attrs are: {df.columns.tolist()}")
            return

        plt.clf()

        plt.title(title)
        g = sns.lineplot(
            x=df.index,
            y=attr,
            data=df,
        )
        for change_point in change_points:
            g.axvline(change_point, ls="--", c="yellow")
        plt.show()
