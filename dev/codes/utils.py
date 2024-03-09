import json
import multiprocessing.pool as mpp

from codes.usp_stream_datasets import load_insect_dataset, insects_datasets


def generate_experiment_filenames(
    dataset: str,
    results_folder: str = "results/",
):
    """Generates experiment filenames."""
    dataset_prefix = results_folder + dataset.lower().replace(".", "").replace(
        "(", ""
    ).replace(")", "").replace(" ", "-")
    csv_file = dataset_prefix + ".csv"
    metadata_file = dataset_prefix + "_metadata.json"
    plot_file = dataset_prefix + ".jpg"
    return csv_file, metadata_file, plot_file


def fetch_original_dataframe(dataset_identifier):
    """Fetches the original dataframe given the identifier from
    usp stream datasets.
    """
    return load_insect_dataset(insects_datasets[dataset_identifier]["filename"])


def open_metadata_file(filename):
    """Open an experiment metadata file."""
    f = open(filename, "r")
    return json.loads(f.read())


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)
