from feature_selection.correlation_metric import PearsonCorrelation, MutualInformation, SignalNoiseRatio, Silhouette
from feature_selection.utils import get_intersection


class BaseSelector(object): pass


class FeatureSelector(BaseSelector):
    def __init__(self, selection_method):
        self.selection_method = selection_method

        if (selection_method == "pearson"):
            self.correlation_metric = PearsonCorrelation()
        elif (selection_method == "mi"):
            self.correlation_metric = MutualInformation()
        elif (selection_method == "snr"):
            self.correlation_metric = SignalNoiseRatio()
        elif (selection_method == "silhouette"):
            self.correlation_metric = Silhouette()
        else:
            self.correlation_metric = None

    def select_features(self, dataset, k, datalabels):
        if not self.correlation_metric:
            print("{} method not known".format(self.selection_method))
            return

        correlations = sorted(self.correlation_metric.get_correlation(dataset, datalabels),
                              key=lambda row: -row[1])[:k]

        return list(list(zip(*correlations))[0])


class SelectionCombinator(BaseSelector):
    def __init__(self, selector_combinations):
        self.feature_selectors = {}
        self.dims = {}
        for selection_method, dims in selector_combinations:
            self.feature_selectors[selection_method] = FeatureSelector(selection_method)
            self.dims[selection_method] = dims

    def _reduce_dimension(self, dataset, selected_features):
        return [[xi[i] for i in selected_features] for xi in dataset]

    def get_reduced_data(self, train_data, validation_data, test_data, train_labels):
        for selection_method in self.feature_selectors:
            feature_selector = self.feature_selectors[selection_method]
            k = self.dims[selection_method]
            selected_features = feature_selector.select_features(train_data, k, train_labels)
            train_data = self._reduce_dimension(train_data, selected_features)
            validation_data = self._reduce_dimension(validation_data, selected_features)
            test_data = self._reduce_dimension(test_data, selected_features)

        return train_data, validation_data, test_data
