import pandas as pd
from scipy.io.arff import loadarff


comparisons_output_dir = "/Users/lucashelfstein/Documents/Masters/drifts/dev/comparison_results"

insects_datasets = {
    "Incremental (bal.)": {
        "change_point": [],
        "filename": "INSECTS-incremental_balanced_norm.arff",
    },
    "Incremental (imbal.)": {
        "change_point": [],
        "filename": "INSECTS-incremental_imbalanced_norm.arff",
    },
    "Abrupt (bal.)": {
        "change_point": [14352, 19500, 33240, 38682, 39510],
        "filename": "INSECTS-incremental-abrupt_balanced_norm.arff",
    },
    "Abrupt (imbal.)": {
        "change_point": [83859, 128651, 182320, 242883, 268380],
        "filename": "INSECTS-incremental-abrupt_imbalanced_norm.arff",
    },
    "Incremental-gradual (bal.)": {
        "change_point": [14028],
        "filename": "INSECTS-gradual_balanced_norm.arff",
    },
    "Incremental-gradual (imbal.)": {
        "change_point": [58159],
        "filename": "INSECTS-gradual_imbalanced_norm.arff",
    },
    "Incremental-abrupt-reoccurring (bal.)": {
        "change_point": [26568, 53364],
        "filename": "INSECTS-abrupt_balanced_norm.arff",
    },
    "Incremental-abrupt-reoccurring (imbal.)": {
        "change_point": [150683, 301365],
        "filename": "INSECTS-abrupt_imbalanced_norm.arff",
    },
    "Incremental-reoccurring (bal.)": {
        "change_point": [26568, 53364],
        "filename": "INSECTS-incremental-reoccurring_balanced_norm.arff",
    },
    "Incremental-reoccurring (imbal.)": {
        "change_point": [150683, 301365],
        "filename": "INSECTS-incremental-reoccurring_imbalanced_norm.arff",
    },
    "Out-of-control": {
        "change_point": [],
        "filename": "INSECTS-out-of-control_norm.arff",
    },
}

file_path_prefix = "/Users/lucashelfstein/Documents/Masters/drifts/dev/data/usp-stream-data/"


def load_insect_dataset(filename: str) -> pd.DataFrame:
    """Read the desired Dataset."""
    raw_data = loadarff(file_path_prefix + filename)
    return pd.DataFrame(raw_data[0])