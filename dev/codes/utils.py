import json


def generate_filenames(dataset: str, results_folder: str = "results/"):
    """Util for experiment filenames."""
    dataset_prefix = results_folder + dataset.lower().replace(".", "").replace(
        "(", ""
    ).replace(")", "").replace(" ", "-")
    csv_file = dataset_prefix + ".csv"
    metadata_file = dataset_prefix + "_metadata.json"
    plot_file = dataset_prefix + ".jpg"
    return csv_file, metadata_file, plot_file


def open_metadata_file(filename):
    """Open an experiment metadata file."""
    f = open(filename, "r")
    return json.loads(f.read())
