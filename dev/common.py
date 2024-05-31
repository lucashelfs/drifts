import os
import pandas as pd

from dev.config import insects_datasets, load_insect_dataset
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

from river_datasets import sea_concept_drift_dataset, stagger_concept_drift_dataset, generate_dataset_from_river_generator, multi_sea_dataset, multi_stagger_dataset

common_datasets = {
    "electricity":
        {
            "filename": "~/Downloads/electricity-normalized.csv",
            "class_column": "class"
        },
    "magic": {
        "filename": "~/Downloads/magic.csv",
        "class_column": "class"
    }
}


def load_magic_dataset_data(file_path="~/Downloads/magic.csv"):
    # Expand the user path
    file_path = os.path.expanduser(file_path)

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the dataset from the file
        df = pd.read_csv(file_path)

    else:
        # Fetch the dataset from the API
        magic_gamma_telescope = fetch_ucirepo(id=159)
        X = magic_gamma_telescope.data.features

        # Convert to DataFrame to keep column names
        df = pd.DataFrame(X, columns=magic_gamma_telescope.data.feature_names)
        df['fConc1'] = df['fConc1'].sort_values().reset_index(drop=True)

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(df)

        # Convert back to DataFrame to retain column names
        scaled_df = pd.DataFrame(scaled_X, columns=df.columns)

        # Save the DataFrame to a CSV file
        scaled_df.to_csv(file_path, index=False)

        # Use the scaled DataFrame as the result
        df = scaled_df

    return df


def load_magic_dataset_targets():
    from ucimlrepo import fetch_ucirepo
    magic_gamma_telescope = fetch_ucirepo(id=159)
    y = magic_gamma_telescope.data.targets
    return y


def load_synthetic_sea(seed, drift_central_position, drift_width, dataset_size):
    sea_generator = sea_concept_drift_dataset(seed, drift_central_position, drift_width, stream_variant=0, drift_variant=1)
    sea_df = generate_dataset_from_river_generator(sea_generator, dataset_size)
    return sea_df


def load_multi_sea(seed, dataset_size):
    msg = multi_sea_dataset(seed)
    multi_sea_df = generate_dataset_from_river_generator(msg, dataset_size)
    return multi_sea_df


def load_synthetic_stagger(seed, drift_central_position, drift_width, dataset_size):
    stagger_generator = stagger_concept_drift_dataset(seed, drift_central_position, drift_width, stream_variant=0, drift_variant=1)
    stagger_df = generate_dataset_from_river_generator(stagger_generator, dataset_size)
    return stagger_df


def load_multi_stagger(seed, dataset_size):
    msg = multi_stagger_dataset(seed)
    multi_stagger_df = generate_dataset_from_river_generator(msg, dataset_size)
    return multi_stagger_df


def load_and_prepare_dataset(dataset):
    # Parameters
    from river_config import seed, drift_central_position, drift_width, dataset_size

    if dataset in insects_datasets.keys():
        dataset_filename_str = (
            dataset.lower()
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )
        X_insects = load_insect_dataset(insects_datasets[dataset]["filename"])
        _ = X_insects.pop("class")
        return X_insects, dataset_filename_str

    elif dataset == "electricity":
        df = pd.read_csv(common_datasets[dataset]['filename'])
        _ = df.pop("class")
        return df, dataset

    elif dataset == "magic":
        return load_magic_dataset_data(), "magic"

    # Synthetic datasets go here
    elif dataset == "SEA":
        df = load_synthetic_sea(seed, drift_central_position, drift_width, dataset_size)
        _ = df.pop("class")
        return df, "SEA"

    elif dataset == "MULTISEA":
        df = load_multi_sea(seed, dataset_size)
        _ = df.pop("class")
        return df, "MULTISEA"

    elif dataset == "STAGGER":
        df = load_synthetic_stagger(seed, drift_central_position, drift_width, dataset_size)
        _ = df.pop("class")
        return df, "STAGGER"

    elif dataset == ("MULTISTAGGER"):
        df = load_multi_stagger(seed, dataset_size)
        _ = df.pop("class")
        return df, "MULTISTAGGER"


def find_indexes(my_list):
    drift_indexes = [index for index, value in enumerate(my_list) if value == "drift"]
    return drift_indexes
