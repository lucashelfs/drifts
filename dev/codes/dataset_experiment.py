from .base_experiment import BaseExperiment


class DatasetExperiment(BaseExperiment):
    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )
