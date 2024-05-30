# This file uses the electricity dataset
# https://www.kaggle.com/datasets/yashsharan/the-elec2-dataset

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score

from dev.config import comparisons_output_dir as output_dir
from dev.common import common_datasets, load_magic_dataset_data, load_magic_dataset_targets, find_indexes, load_synthetic_sea, load_synthetic_stagger, load_multi_sea, load_multi_stagger
from dev.ddm import fetch_ksddm_drifts, fetch_hdddm_drifts, fetch_jsddm_drifts
from dev.config import insects_datasets, load_insect_dataset
from dev.ksddm_tester_dry import define_batches


def run_prequential_naive_bayes(dataset="Abrupt (bal.)", batch_size=1000, batches_with_drift_list=None):
    reference_batch = 1

    # TODO: refactor the data import
    from river_config import seed, drift_central_position, drift_width, dataset_size

    if dataset in insects_datasets.keys():
        X = load_insect_dataset(insects_datasets[dataset]["filename"])
        Y_og = X.pop("class")
    else:
        if dataset == "electricity":
            X = pd.read_csv(common_datasets[dataset]['filename'])
            Y_og = X.pop(common_datasets[dataset]['class_column'])
        elif dataset == "magic":
            X = load_magic_dataset_data()
            Y_og = load_magic_dataset_targets()
            Y_og = Y_og.values.ravel()
        elif dataset == "SEA":
            X = load_synthetic_sea(seed, drift_central_position, drift_width, dataset_size)
            Y_og = X.pop("class")
        elif dataset == "MULTISEA":
            X = load_multi_sea(seed, dataset_size)
            Y_og = X.pop("class")
        elif dataset == "STAGGER":
            X = load_synthetic_stagger(seed, drift_central_position, drift_width, dataset_size)
            Y_og = X.pop("class")
        elif dataset == "MULTISTAGGER":
            X = load_multi_stagger(seed, dataset_size)
            Y_og = X.pop("class")
        else:
            raise Exception("Wrong dataset configuration")

    le = LabelEncoder()
    le.fit(Y_og)
    Y = le.transform(Y_og)
    Y = pd.DataFrame(Y, columns=["class"])

    X = define_batches(X=X, batch_size=batch_size)
    Y = define_batches(X=Y, batch_size=batch_size)

    reference = X[X.Batch == reference_batch].iloc[:, :-1]
    y_reference = Y[Y.Batch == reference_batch].iloc[:, :-1].values.ravel()

    base_classifier = MultinomialNB()
    base_classifier.partial_fit(reference, y_reference, np.unique(Y['class']))

    batches = [batch for batch in X.Batch if batch != reference_batch]
    batches = list(set(batches))

    batch_predictions = []
    drift_indexes = []

    y_pred = base_classifier.predict(reference)
    batch_predictions.append(y_pred)

    if batches_with_drift_list is not None:
        drift_indexes = [index + 1 for index, value in enumerate(batches_with_drift_list) if value == "drift"]

    for batch in batches:
        X_batch = X[X.Batch == batch].iloc[:, :-1]
        Y_batch = Y[Y.Batch == batch].iloc[:, :-1].values.ravel()

        if batches_with_drift_list is not None:
            if batch in drift_indexes:
                reference = X_batch
                y_reference = Y_batch
                base_classifier = MultinomialNB()
                base_classifier.partial_fit(reference, y_reference, np.unique(Y['class']))
            else:
                base_classifier.partial_fit(X_batch, Y_batch)
        else:
            base_classifier.partial_fit(X_batch, Y_batch)

        y_pred = base_classifier.predict(X_batch)
        batch_predictions.append(y_pred)

    batch_predictions = [item for sublist in batch_predictions for item in sublist]
    return X, Y, batch_predictions


def run_test(dataset, batch_size=1000, plot_heatmaps=True):
    ks_drifts = fetch_ksddm_drifts(batch_size=batch_size, dataset=dataset, plot_heatmaps=plot_heatmaps, text="KSDDM 95")
    ks_90_drifts = fetch_ksddm_drifts(batch_size=batch_size, dataset=dataset, plot_heatmaps=plot_heatmaps, mean_threshold=0.10, text="KSDDM 90")
    hd_drifts = fetch_hdddm_drifts(batch_size=batch_size, plot_heatmaps=plot_heatmaps, dataset=dataset)
    js_drifts = fetch_jsddm_drifts(batch_size=batch_size, plot_heatmaps=plot_heatmaps, dataset=dataset)

    X, Y, batch_predictions_base = run_prequential_naive_bayes(dataset=dataset, batch_size=batch_size)
    X, Y, batch_predictions_ks = run_prequential_naive_bayes(dataset=dataset, batch_size=batch_size, batches_with_drift_list=ks_drifts)
    X, Y, batch_predictions_ks_90 = run_prequential_naive_bayes(dataset=dataset, batch_size=batch_size, batches_with_drift_list=ks_90_drifts)
    X, Y, batch_predictions_hd = run_prequential_naive_bayes(dataset=dataset, batch_size=batch_size, batches_with_drift_list=hd_drifts)
    X, Y, batch_predictions_js = run_prequential_naive_bayes(dataset=dataset, batch_size=batch_size, batches_with_drift_list=js_drifts)

    y_true = Y['class'].values

    results = {
        'KS95_drifts': find_indexes(ks_drifts.tolist()),
        'KS90_drifts': find_indexes(ks_90_drifts.tolist()),
        'HD_drifts': find_indexes(hd_drifts.tolist()),
        'JS_drifts': find_indexes(js_drifts.tolist())
    }

    metrics_results = {}

    # Calculate metrics
    for name, predictions in zip(['Base', 'KS95', 'KS90', 'HD', 'JS'], [batch_predictions_base, batch_predictions_ks, batch_predictions_ks_90, batch_predictions_hd, batch_predictions_js]):
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, predictions, pos_label=1)
        roc_auc = auc(fpr, tpr)

        metrics_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_curve': (fpr, tpr, roc_auc)
        }

        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

    return results, metrics_results


def plot_drift_points(drift_results, dataset, batch_size):
    os.makedirs(output_dir + f"/{dataset}/detected_drifts/", exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = {'KS95_drifts': 'r', 'KS90_drifts': 'g', 'HD_drifts': 'b', 'JS_drifts': 'm'}

    # Ensure all methods are on the Y-axis
    methods = ['KS95_drifts', 'KS90_drifts', 'HD_drifts', 'JS_drifts']
    for method in methods:
        drifts = drift_results.get(method, [])
        plt.scatter(drifts, [method] * len(drifts), color=colors[method], label=method if len(drifts) > 0 else None,
                    s=50)

    plt.xlabel('Batch Index')
    plt.ylabel('Detection Method')
    plt.title(f'Drift Points for {dataset} (Batch Size: {batch_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir + f"/{dataset}/detected_drifts/", f'{dataset}_{batch_size}_drift_points.png'))
    plt.close()


def plot_results(results, dataset, batch_sizes):
    os.makedirs(output_dir + f"/{dataset}/metrics/", exist_ok=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Create separate plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model in results[batch_sizes[0]]:
            metric_values = [results[batch_size][model][metric] for batch_size in batch_sizes]
            plt.plot(batch_sizes, metric_values, marker='o', label=model)
        plt.xlabel('Batch Size')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison for {dataset}')
        plt.legend()
        plt.grid(True)
        plt.xticks(batch_sizes)
        plt.savefig(os.path.join(output_dir + f"/{dataset}/metrics/", f'{metric}.png'))
        plt.close()

    # Plot ROC Curve for each batch size in separate files
    for batch_size in batch_sizes:
        plt.figure(figsize=(10, 6))
        for model in results[batch_size]:
            fpr, tpr, roc_auc = results[batch_size][model]['roc_curve']
            plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {dataset} (Batch Size: {batch_size})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir + f"/{dataset}/metrics/", f'roc_curve_{batch_size}.png'))
        plt.close()

    # Collect all ROC curves for top 3 plot
    roc_curves = []
    for batch_size in batch_sizes:
        for model in results[batch_size]:
            fpr, tpr, roc_auc = results[batch_size][model]['roc_curve']
            roc_curves.append((fpr, tpr, roc_auc, f'{model} (Batch Size: {batch_size})'))

    # Sort ROC curves by AUC in descending order and plot the top 3
    roc_curves.sort(key=lambda x: x[2], reverse=True)
    plt.figure(figsize=(10, 6))
    for fpr, tpr, roc_auc, label in roc_curves[:3]:
        plt.plot(fpr, tpr, label=f'{label}, AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Top 3 ROC Curves for {dataset}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir + f"/{dataset}/metrics/", f'top3_roc_curve.png'))
    plt.close()


def save_results_to_csv(dataset, batch_size, drift_results, metrics_results, csv_file_path):
    # Create the data structure to be saved in CSV
    data = []
    for technique, metrics in metrics_results.items():
        if technique == 'Base':
            num_drifts = 0
        else:
            num_drifts = len(drift_results[f"{technique}_drifts"])

        fpr, tpr, roc_auc = metrics['roc_curve']
        fpr_str = ';'.join(map(str, fpr))
        tpr_str = ';'.join(map(str, tpr))

        data.append({
            'dataset': dataset,
            'batch_size': batch_size,
            'technique': technique,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'num_drifts': num_drifts,
            # 'fpr': fpr_str,
            # 'tpr': tpr_str,
            'auc': roc_auc
        })

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Append the DataFrame to the CSV file
    df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
    print(f"Results for {dataset}, batch size {batch_size} saved to {csv_file_path}")


def consolidate_csv_files(csv_file_paths, target_csv_file):
    # List to hold the data frames
    df_list = []

    # Read each CSV file and append it to the list
    for file in csv_file_paths:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all data frames into a single data frame
    consolidated_df = pd.concat(df_list, ignore_index=True)

    # Save the consolidated data frame to the target CSV file
    consolidated_df.to_csv(target_csv_file, index=False)
    print(f"Consolidated results saved to {target_csv_file}")


def main():

    batch_sizes = [1000, 1500, 2000, 2500]
    datasets = ["MULTISTAGGER", "MULTISEA", "SEA", "STAGGER", "electricity", "magic"]
    for dataset in insects_datasets.keys():
        if dataset != "Out-of-control":
            datasets.append(dataset)

    results = {dataset: {} for dataset in datasets}
    csv_file_paths = []  # List to store the paths of the generated CSV files

    for dataset in datasets:
        dataset_results = {}
        output_path = os.path.join(output_dir, f"{dataset}")
        os.makedirs(output_path, exist_ok=True)
        csv_file_path = os.path.join(output_path, f"{dataset}_results.csv")

        # Remove the file if it already exists
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        for batch_size in batch_sizes:
            print(f"{dataset} - {batch_size}")
            drift_results, test_results = run_test(dataset=dataset, batch_size=batch_size, plot_heatmaps=True)
            dataset_results[batch_size] = test_results
            plot_drift_points(drift_results, dataset, batch_size)
            save_results_to_csv(dataset, batch_size, drift_results, test_results, csv_file_path)
            print()

        results[dataset] = dataset_results
        plot_results(dataset_results, dataset, batch_sizes)
        csv_file_paths.append(csv_file_path)  # Add the CSV file path to the list

    # Consolidate all CSV files into a single CSV file
    target_csv_file = os.path.join(output_dir, "consolidated_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    consolidate_csv_files(csv_file_paths, target_csv_file)


if __name__ == "__main__":
    main()
