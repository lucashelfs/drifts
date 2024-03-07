from config import DEFAULT_RESULTS_FOLDER
from abc import ABC

TEST_TYPES = ["ks", "kl_dummy", "kl_median", "js_dummy", "js_median"]


class BaseExperiment(ABC):
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

        # TODO: develop synthetic data logic and handling inside the class
        self.data_source = kwargs.get("data_source", False)
        if not self.data_source:
            raise ("The data source must be specified. Available: insects.")

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