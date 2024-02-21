from binning import (
    simple_binning,
    plot_binning,
    calculate_kl_with_dummy_bins,
    calculate_kl_with_median,
    calculate_js_with_median,
)

import os


from experiment import Experiment

results_folder = os.getcwd() + "/results_of_kl_experiments/"
dataset_name = "Incremental (bal.)"

exp = Experiment(
    dataset=dataset_name,
    batches=False,
    results_folder=results_folder,
    train_percentage=0.25,
)
exp.prepare_insects_test()

attr = "Att31"
start = 0
end = (
    start + exp.df_baseline.shape[0]
)  # ACHO que não tem mais essa restrição aqui do tamanho -> embora se for mto pequeno a KL dara inf -> mas isso deve depender mto diretamente dos dados

# ha uma relacao entre o numero de bins a ser usado e a loucura da dist dos dados

baseline = exp.df_baseline[attr]
stream = exp.df_stream[attr][start:end]


plot_binning(
    baseline, stream, binning_type="median", median_origin="both", n_bins=10
)


# Calculate KL divergence: dummy
n_bins = 3
kl_divergence = calculate_kl_with_dummy_bins(baseline, stream, n_bins)
print(n_bins, kl_divergence)


# Calculate KL divergence: median both
kl_divergence = calculate_kl_with_median(
    baseline, stream, median_origin="both", n_bins=n_bins
)
print(n_bins, kl_divergence)


# Calculate JensonShannon divergence: median both
js_distance = calculate_js_with_median(
    baseline, stream, median_origin="both", n_bins=n_bins
)
print(n_bins, js_distance)
