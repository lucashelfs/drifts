from .base_experiment import BaseExperiment


class SyntheticExperiment(BaseExperiment):
    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

    def set_dataframes(self):
        """Abstract method for handling the baseline and stream dataframes."""
        raise NotImplementedError("This method has not been implemented.")
