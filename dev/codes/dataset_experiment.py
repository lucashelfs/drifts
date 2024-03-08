from .base_experiment import BaseExperiment


class DatasetExperiment(BaseExperiment):
    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

        self.write_dataset_metadata()

    def set_data_source(self):
        self.data_source = "dataset"

    def print_experiment_info(self):
        """Print experiment information about the dataset being used."""
        super().print_experiment_info()
        print("DATASET SPECIFIC INFORMATION")
        print(f"DF Total: {self.total_df.shape}")

    def write_dataset_metadata(self):
        """Write experiment metadata on its data structure."""
        self.metadata["dataset"] = self.dataset
