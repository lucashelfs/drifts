# Entities of the experiments

# AS IS

```mermaid
classDiagram

Visualizer --|> Experiment

class Experiment{
    +String id
    +String type
    +SmallDecimal p_value

    +set_dataset_prefix()
    +validate_given_attr()
    +write_initial_metadata()
    +set_result_directory()
    +load_insects_dataframe()
    +fetch_classes_and_minimal_class()
    +set_window_size()
    +create_baseline_dataframe()
    +create_stream_dataframe()
    +print_experiment_dfs()
    +write_metadata()
    +mp_window_test()
    +async_test()
    +set_attr_cols()
    +async_test_for_desired_attrs()
    +prepare_insects_test()
    +run_insects_test()
}

class Visualizer{
    +String experiment_id
}

```

***

# TO BE


```mermaid
classDiagram


class Experiment{
    +String id
    +String test_type

    +set_attr_cols()
    +set_dataset_prefix()
    +set_result_directory()

    +write_metadata(initial=False)
    +write_results()
    
    +create_baseline_dataframe()
    +create_stream_dataframe()
    
    +mp_window_test()
    +async_test()
    +set_test_window_size()
}

class MultiStreamExperiment{
    +Dataframe Test
}

class SingleStreamExperiment{
    +validate_given_attr()
    +async_test_for_desired_attr()
}

class DatasetExperiment{
    +Dataframe Baseline
    +Dataframe Stream
    +fetchData()
}

class SyntheticExperiment{
    +fetchSyntheticData()
}

class insectsExperiment{
    +load_insects_dataframe()
    +prepare_insects_test()
    +run_insects_test()
    +fetch_classes_and_minimal_class()
}

class Visualizer{
    +String experiment_id
}

SyntheticExperiment --|> SingleStreamExperiment 
SingleStreamExperiment --|> Experiment

insectsExperiment --|> DatasetExperiment
DatasetExperiment --|> MultiStreamExperiment
MultiStreamExperiment --|> Experiment

Visualizer --|> Experiment



```