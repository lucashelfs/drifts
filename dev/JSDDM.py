from dev.histogram_density_method_with_ks_and_js import HistogramDensityMethodWithKSAndJS


class JSDDM(HistogramDensityMethodWithKSAndJS):
    input_type = "batch"

    def __init__(
        self,
        detect_batch=1,
        divergence="JS",
        statistic="tstat",
        significance=0.05,
        subsets=5,
    ):

        super().__init__(
            divergence=divergence,
            detect_batch=detect_batch,
            statistic=statistic,
            significance=significance,
            subsets=subsets,
        )

    def update(self, X, y_true=None, y_pred=None):
        super().update(X, y_true, y_pred)
