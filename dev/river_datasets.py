# Using river datasets

from river.datasets import synth
import pandas as pd
import matplotlib.pyplot as plt


def generate_dataset_from_river_generator(dataset, dataset_size):
    if not isinstance(dataset, list):
        # Initialize lists to store the data
        data = []
        labels = []

        # Generate and store the dataset
        for x, y in dataset.take(dataset_size):
            data.append(x)
            labels.append(y)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['class'] = labels
        return df
    else:
        # Check if the dataset list is empty
        if len(dataset) == 0:
            raise ValueError("The dataset list is empty.")

        # Initialize lists to store the data
        all_data = []
        dataset_size_chunked = dataset_size // len(dataset)
        remaining_size = dataset_size % len(dataset)

        for d in dataset:
            data = []
            labels = []

            # Adjust the chunk size to account for any remaining size
            current_chunk_size = dataset_size_chunked + (1 if remaining_size > 0 else 0)
            remaining_size -= 1

            # Generate and store the dataset
            for x, y in d.take(current_chunk_size):
                data.append(x)
                labels.append(y)

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['class'] = labels
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)


def sea_concept_drift_dataset(seed,  drift_central_position, drift_width, stream_variant=0, drift_variant=3):
    # Create a synthetic dataset with concept drift using SEA as a base stream
    sea_dataset = synth.ConceptDriftStream(
        stream=synth.SEA(seed=seed, variant=stream_variant, noise=0.05),
        drift_stream=synth.SEA(seed=seed, variant=drift_variant, noise=0.05),
        seed=seed, position=drift_central_position, width=drift_width
    )
    return sea_dataset


def multi_sea_dataset(seed):
    # Create a synthetic dataset with concept drift using SEA as a base stream
    return [synth.SEA(seed=seed, variant=variant, noise=0.05) for variant in range(4)]


def multi_stagger_dataset(seed):
    # Create a synthetic dataset with concept drift using SEA as a base stream
    return [synth.STAGGER(seed=seed, classification_function=variant) for variant in range(3)]


def stagger_concept_drift_dataset(seed,  drift_central_position, drift_width, stream_variant=0, drift_variant=1):
    # Create a synthetic dataset with concept drift using STAGGER as a base stream
    stagger_dataset = synth.ConceptDriftStream(
        stream=synth.STAGGER(classification_function=stream_variant, seed=seed, balance_classes=False),
        drift_stream=synth.STAGGER(classification_function=drift_variant, seed=seed, balance_classes=False),
        seed=seed, position=drift_central_position, width=drift_width
    )
    return stagger_dataset


def plot_dataset(df, title, drift_start=None, drift_end=None):

    # Plot each feature and the label in separate plots
    num_features = len(df.columns) - 1  # Excluding the label column
    num_plots = num_features + 1  # Including the label plot
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 4), sharex=True)

    # Plot each feature
    for i, col in enumerate(df.columns):
        if col == 'class':
            axes[i].scatter(df.index, df[col], color='red', s=10, label=col)
        else:
            axes[i].scatter(df.index, df[col], label=col)

        if drift_start:
            axes[i].axvline(x=drift_start, color='green', linestyle='--', label='Drift Start')
        if drift_end:
            axes[i].axvline(x=drift_end, color='green', linestyle='--', label='Drift End')

        axes[i].set_title(f'{col} over time')
        axes[i].set_ylabel(col)
        axes[i].legend()

    # Set x-axis label for the last subplot
    axes[-1].set_xlabel('Order of rows')

    # Set the main title
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()
    plt.close()


def main():

    # Parameters
    from river_config import seed, drift_central_position, drift_width, dataset_size
    drift_start = drift_central_position - drift_width // 2
    drift_end = drift_central_position + drift_width // 2

    msg = multi_stagger_dataset(seed)
    multi_stagger_df = generate_dataset_from_river_generator(msg, dataset_size)
    plot_dataset(multi_stagger_df, 'Multi STAGGER dataset')

    msg = multi_sea_dataset(seed)
    multi_sea_df = generate_dataset_from_river_generator(msg, dataset_size)
    plot_dataset(multi_sea_df, 'Multi SEA dataset')

    sea_generator = sea_concept_drift_dataset(seed, drift_central_position, drift_width, stream_variant=0, drift_variant=1)
    sea_df = generate_dataset_from_river_generator(sea_generator, dataset_size)
    plot_dataset(sea_df, 'SEA Dataset with Concept Drift', drift_start, drift_end)

    stagger_generator = stagger_concept_drift_dataset(seed, drift_central_position, drift_width, stream_variant=0, drift_variant=1)
    stagger_df = generate_dataset_from_river_generator(stagger_generator, dataset_size)
    plot_dataset(stagger_df, 'STAGGER Dataset with Concept Drift', drift_start, drift_end)


if __name__ == "__main__":
    main()
