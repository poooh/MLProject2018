from math import sqrt

class Statistics:
    @classmethod
    def mean(cls, dataset):
        l = len(dataset)

        if isinstance(dataset[0], list):
            return [sum(dim) / l for dim in zip(*dataset)]

        return (sum(dataset) / l)

    @classmethod
    def variance(cls, dataset, mean_data=None):
        if mean_data is None:
            mean_data = cls.mean(dataset)

        l = len(dataset)
        if isinstance(dataset[0], list):
            return [sum([(i-m) ** 2 for i in xi])/l for xi, m in zip(zip(*dataset), mean_data)]

        return sum((xi - mean_data) ** 2 for xi in dataset) / l

    @classmethod
    def covariance(cls, dataset_u, dataset_v, mean_u=None, mean_v=None):
        if mean_u is None:
            mean_u = cls.mean(dataset_u)
        if mean_v is None:
            mean_v = cls.mean(dataset_v)

        return sum((xi - mean_u) * (yi - mean_v) for xi, yi in zip(dataset_u, dataset_v)) / len(dataset_v)

    @classmethod
    def std_dev(cls, dataset, mean_data=None):
        if isinstance(dataset[0], list):
            return [sqrt(var) for var in cls.variance(dataset, mean_data)]

        return sqrt(cls.variance(dataset, mean_data))



class Evaluation:
    @classmethod
    def _01_accuracy(cls, predictions, true_labels):
        return sum(1.0 if yi == ri else 0.0 for yi, ri in zip(predictions, true_labels)) / len(predictions)

    @classmethod
    def get_accuracy(cls, predictions, true_labels, scoring = "0-1"):
        if isinstance(scoring, str):
            if scoring == "0-1":
                scoring = cls._01_accuracy

        return scoring(predictions, true_labels)


class LinAlg:
    @classmethod
    def distance(cls, u, v):
        if isinstance(u, float):
            return abs(u - v)
        return sqrt(sum((ui - vi) ** 2 for ui, vi in zip(u, v)))

