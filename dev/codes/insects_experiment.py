from .dataset_experiment import DatasetExperiment


class InsectsExperiment(DatasetExperiment):
    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

    def validate_given_attr(self):
        """Validate single attribute inputs."""
        if self.test_type == "insects":
            if not self.attr.startswith("Att"):
                raise ("The given attribute doesn't match the insects DF pattern.")

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
