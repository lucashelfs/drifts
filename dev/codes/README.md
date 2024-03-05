# Entities of the experiments



```mermaid

classDiagram
class Experiment{
    +String id
    +String type
    +BigDecimal p_value

}

class Visualizer{
    +String experiment_id
}

Visualizer --|> Experiment

```

