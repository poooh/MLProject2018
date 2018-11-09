from utils import NaiveBayesHelper
from base import BaseLearner, Mapper


class NaiveBayesClassifier(BaseLearner):
    def __init__(self, classes=-1):
        '''
        NAIVE BAYES CLASSIFIER INSTANCE

        :param n: int, Number of classes. Defaults to -1 which means number of classes not known.
        :coeff _class_means: list, Array of class means. Shape = (n_classes, n_features).
        :coeff _class_std_dev: list, Array of class standard deviations. Shape = (n_classes, n_features).
        :coeff _mapper: Mapper, Mapper object to map labels to integer classes from 0 to n_classes-1
        '''
        self.n_classes = classes
        self._class_means = None
        self._class_std_dev = None
        self._mapper = None

    def fit(self, X, y):
        '''
        FUNCTION TO FIT THE TRAINING DATA IN CLASSIFIER

        :param X: list, Features array. shape = (n_samples, n_features).
        :param y: list, Labels array. shape = (n_sapmples,)
        :return: None
        '''
        if not (type(X) is list and type(X[0]) is list):
            print("Features should be 2-D list of shape (n_samples, n_features)")
            return
        if type(y) is not list:
            print("Labels should be a list")
            return

        if not self._mapper:
            self._mapper = Mapper(y)
        y = self._mapper.map_labels(y)
        self.n_classes = len(self._mapper.label_map.keys())

        self._class_means, self._class_std_dev = NaiveBayesHelper.get_mean_std_dev(X, y, self.n_classes)

    def predict(self, X):
        '''
        FUNCTION TO PREDICT LABELS FOR TEST DATA

        :param X: list, Features array. shape = (n_samples, n_features).
        :return:
        '''
        if not self._class_means:
            return None
        if not (type(X) is list and type(X[0]) is list):
            print("Test Features should be 2-D list of shape (n_samples, n_features)")
            return None

        predicted_labels = []
        for x in X:
            distances_0 = NaiveBayesHelper.get_normalized_vector_distance(x, self._class_means[0],
                                                                          self._class_std_dev[0])
            distances_1 = NaiveBayesHelper.get_normalized_vector_distance(x, self._class_means[1],
                                                                          self._class_std_dev[1])

            prediction = 0 if distances_0 <= distances_1 else 1

            predicted_labels.append(prediction)

        return self._mapper.map_reverse(predicted_labels)
