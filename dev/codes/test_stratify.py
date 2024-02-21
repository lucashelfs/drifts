from experiment import Experiment
from usp_stream_datasets import insects_datasets


for dataset_name in insects_datasets.keys():
    exp = Experiment(dataset=dataset_name)
    exp.prepare_insects_test()
    dataset = exp.total_df
    print(f"{dataset_name} - {dataset.shape}")
    print(dataset["class"].value_counts())


SEED = 123
BASELINE_PROPORTION = 0.2


for dataset_name in insects_datasets.keys():
    exp = Experiment(dataset=dataset_name)
    exp.prepare_insects_test()
    dataset = exp.total_df
    print(f"{dataset_name} - {dataset.shape}")

    strata_df = exp.total_df.groupby("class", group_keys=False).apply(
        lambda x: x.sample(frac=0.2, random_state=SEED)
    )
    print(dataset_name)
    print(strata_df["class"].value_counts())


#########


from experiment import Experiment
from usp_stream_datasets import insects_datasets


SEED = 123
BASELINE_PROPORTION = 0.2


for dataset_name in insects_datasets.keys():
    exp = Experiment(dataset=dataset_name)
    exp.prepare_insects_test()
    dataset = exp.total_df
    strata_df = exp.total_df.groupby("class", group_keys=False).apply(
        lambda x: x.sample(frac=0.2, random_state=SEED)
    )
    print(dataset_name)
    print(strata_df["class"].value_counts())


# stratified_sample = dataset.groupby("class").apply(
#     lambda x: x.sample(frac=0.20, random_state=SEED)
# )

# stratified_sample.head()


# # Remove the extra index added by groupby()
# stratified_sample = stratified_sample.droplevel(0)


# # Ratio of selected items by the island
# stratified_ratio = stratified_sample["class"].value_counts(normalize=True)

# # Convert to percentage
# stratified_ratio = stratified_ratio.round(4) * 100

# # We did stratified sampling. So give it proper name
# stratified_ratio.name = "Stratified"

# # Add it to the variable island_ratios which already has
# # the  expected and SRS proportions

# import pandas as pd
# island_ratios = pd.concat([island_ratios, stratified_ratio], axis=1)
# island_ratios


# df_sample = dataset.groupby("class", group_keys=False).apply(
#     lambda x: x.sample(frac=0.6)
# )
