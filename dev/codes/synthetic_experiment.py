import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from codes.base_experiment import BaseExperiment
from codes.synthetic_data import SyntheticData


class SyntheticExperiment(BaseExperiment):
    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

        self.synthetic_generator = SyntheticData(**kwargs)
        self.window_size = kwargs.get("window_size", False)
        self.visualize_stream = kwargs.get("visualize_stream", False)

        if not self.window_size:
            raise Exception("Missing something...")

        self.set_dataframes(**kwargs)
        self.set_test()
        self.set_change_points()

    def set_change_points(self):
        """Set the change points to be plotted on the visualization."""
        change_points = self.synthetic_generator.change_points
        change_points_dict = {"baseline": [], "stream": []}
        for change_point in change_points:
            if change_point < len(self.df_baseline):
                change_points_dict["baseline"].append(change_point)
            else:
                change_points_dict["stream"].append(change_point - len(self.df_baseline))
        self.change_points = change_points_dict

    def set_dataframes(self, **kwargs):
        """Abstract method for handling the baseline and stream dataframes."""

        if self.visualize_stream:
            self.synthetic_generator.plot_stream()

        self.stream = self.synthetic_generator.get_stream()
        self.create_baseline()
        self.create_stream()

    def create_baseline(self):
        """Create a baseline stream dataframe for the experiment."""
        self.baseline = self.stream[: self.window_size]
        self.df_baseline = (
            pd.DataFrame(self.baseline, columns=["value"])
            .rename_axis("original_index")
            .reset_index()
        )

    def create_stream(self):
        """Create a synthethic stream dataframe for the experiment."""
        self.df_stream = (
            pd.DataFrame(self.stream[self.window_size :], columns=["value"])
            .rename_axis("original_index")
            .reset_index()
        )

    def set_attr_cols(self):
        """Set the atribute columns for the experiment."""
        self.desired_cols = ["value"]

    def set_test(self):
        """Set what else is needed for the test."""
        self.set_attr_cols()

    def set_data_source(self):
        self.data_source = "synthetic"

    def set_results_dataset_filename(self):
        self.dataset_prefix = self.results_folder + "test"

    def print_experiment_info(self):
        super().print_experiment_info()
        print("SYNTHETIC INFORMATION")
        print(f"DF baseline: {self.df_baseline.shape}")
        print(f"DF stream: {self.df_stream.shape}")

    def run_test(self):
        """Logic for a run of the insect experiment."""
        self.print_experiment_info()
        self.async_test()


class SynthethicVisualizer:
    """Visualizer for the results of a synthetic experiment."""

    sns.set(rc={"figure.figsize": (12, 6)})

    def __init__(self, experiment) -> None:
        assert isinstance(experiment, SyntheticExperiment)
        self.experiment = experiment

    def fetch_experiment_filename(self):
        """Create the output filenames from the given experiment."""
        results_file = self.experiment.dataset_prefix + ".csv"
        metadata_file = self.experiment.dataset_prefix + "_metadata.json"
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
                "Attribute not defined for result values. Using default from experiment."
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

        if df_plot.empty:
            print(
                "The dataframe for plotting is empty! The full analysis will be plotted."
            )
            df_plot = df_analysis

        n_bins = kwargs.get("n_bins", False)
        median_origin = kwargs.get("median_origin", False)

        if not n_bins:
            title = f"Distance for synthetic stream indexed by {index} - {attr} - Test type: {test_type}"
        else:
            title = f"Distance for synthetic stream indexed by {index} - {attr} - bins: {n_bins} - Test type: {test_type}"

        if median_origin:
            title = title + f" - Median Origin: {median_origin}"

        plt.clf()
        plt.title(title)
        g = sns.lineplot(x=index, y=result_col, hue="attr", data=df_plot)

        change_points = self.experiment.change_points
        stream_change_points = change_points["stream"]
        if stream_change_points:
            for change_point in stream_change_points:
                g.axvline(change_point, ls="--", c="yellow")

        if savefig:
            plt.savefig(plot_file)

        plt.show()
