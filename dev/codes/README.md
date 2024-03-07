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
    -String experiment_path

    -fetch_change_points()
    -fetch_experiment_change_points()
    -plot_p_values()
    -plot_dataset_change_points()
    -plot_original_data()
    -fetch_top_n_accepted_attributes()
    -plot_multiple_p_values()
    -plot_kl_values()
}

class BinningVisualizer{
    +Dataframe baseline
    +Dataframe stream

    -plot_binning()
    -create_binning_frame()
    -create_binning_timeline()
}

```

***

# TO BE


```mermaid
classDiagram

class Experiment{
    -String id
    -String test_type
    -String results_path
    -Dataframe baseline
    -Dataframe stream
    
    +set_attr_cols()
    +set_baseline_dataframe()
    +set_stream_dataframe()
    +set_dataset_prefix()
    +set_result_directory()

    +write_metadata(initial=False)
    +write_results() 
    
    +async_test()
    +mp_window_test()
    +set_test_window_size()

    +validate_desired_attrs()
    +run_async_test_for_attrs()
}

class DatasetExperiment{
    +String dataset_path

    -get_data_from_file()
    -set_baseline_dataframe()
    -set_stream_dataframe()

}

class SyntheticExperiment{
    +String synthetic_stream_creator_method
    
    -get_synthetic_data()
    -set_baseline_dataframe()
    -set_stream_dataframe()
}

class insectsExperiment{
    -fetch_classes_and_minimal_class()
    -load_insects_dataframe()
    -prepare_insects_test()
    -run_insects_test()
}

class Visualizer{
    -String experiment_path

    -fetch_change_points()
    -fetch_experiment_change_points()
    -plot_p_values()
    -plot_dataset_change_points()
    -plot_original_data()
    -fetch_top_n_accepted_attributes()
    -plot_multiple_p_values()
    -plot_kl_values()
}

class BinningVisualizer{
    +Dataframe baseline
    +Dataframe stream

    -plot_binning()
    -create_binning_frame()
    -create_binning_timeline()
}


SyntheticExperiment --|> Experiment

insectsExperiment --|> DatasetExperiment
DatasetExperiment --|> Experiment

Visualizer --|> Experiment
BinningVisualizer --|> Experiment


```