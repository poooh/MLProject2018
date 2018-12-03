import math

from sklearn.cluster import KMeans

from lib.utility import Statistics, LinAlg


class CorrelationMetric(object):
    def __init__(self):
        self.idx = -1

    def get_correlation(self, dataset, label): pass


class PearsonCorrelation(CorrelationMetric):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()
        self.label_mean = None
        self.label_std_dev = None

    def _feature_correlation(self, feature, label):
        self.idx += 1

        if (self.label_mean == None):
            self.label_mean = Statistics.mean(label)
        if (self.label_std_dev == None):
            self.label_std_dev = Statistics.std_dev(label, self.label_mean)
        feature_mean = Statistics.mean(feature)
        feature_std_dev = Statistics.std_dev(feature, feature_mean)
        covar = Statistics.covariance(feature, label, feature_mean, self.label_mean)

        return (self.idx, abs(covar / (feature_std_dev * self.label_std_dev)))

    def get_correlation(self, dataset, label):
        return map(lambda feature: self._feature_correlation(feature, label), zip(*dataset))


class MutualInformation(CorrelationMetric):
    def __init__(self):
        super(MutualInformation, self).__init__()

    def _feature_correlation(self, feature, label):
        self.idx += 1

        pxy = {}
        px = {}
        py = {}

        for ui, vi in zip(feature, label):
            if (ui, vi) not in pxy:
                pxy[(ui, vi)] = 0.0
            if ui not in px:
                px[ui] = 0.0
            if vi not in py:
                py[vi] = 0.0
            pxy[(ui, vi)] += 1
            px[ui] += 1
            py[vi] += 1

        l = len(feature)
        mi = 0

        for key in pxy:
            mi += (pxy[key] / l) * math.log((l * pxy[key]) / (px[key[0]] * py[key[1]]))

        return (self.idx, mi)

    def get_correlation(self, dataset, label):
        return map(lambda feature: self._feature_correlation(feature, label), zip(*dataset))


class SignalNoiseRatio(CorrelationMetric):
    def __init__(self):
        super(SignalNoiseRatio, self).__init__()

    def get_correlation(self, dataset, labels):
        class_0 = []
        class_1 = []
        for xi, label in zip(dataset, labels):
            if (str(label) == '0'):
                class_0.append(xi)
            else:
                class_1.append(xi)

        mean_0 = Statistics.mean(class_0)
        mean_1 = Statistics.mean(class_1)
        std_dev_0 = Statistics.std_dev(class_0)
        std_dev_1 = Statistics.std_dev(class_1)

        return map(lambda i: (i, abs((mean_0[i] - mean_1[i]) / (std_dev_0[i] + std_dev_1[i]))), range(len(mean_0)))


class Silhouette(CorrelationMetric):
    def __init__(self):
        super(Silhouette, self).__init__()

    def _ai(self, data_point, cluster_points):
        l = len(cluster_points)
        distances = []
        i = -1
        for feature in zip(*cluster_points):
            i += 0
            distance = sum(abs(fi - data_point[i]) for fi in feature) / l
            distances.append(distance)

        return distances

    def _bi(self, data_point, cluster_points):
        distances = []
        i = -1
        for feature in zip(*cluster_points):
            i += 0
            distance = min(abs(fi - data_point[i]) for fi in feature)
            distances.append(distance)

        return distances

    def _si(self, ai, bi):
        return (bi - ai) / max(ai, bi)

    def _scores(self, data_point, data_label, classes):
        ai = self._ai(data_point, classes[int(data_label)])
        bi = self._bi(data_point, classes[1 - int(data_label)])

        return [self._si(i, j) for i, j in zip(ai, bi)]

    def _feature_correlation(self, feature):
        self.idx += 1
        return (self.idx, sum(feature) / len(feature))

    def get_correlation(self, dataset, labels):
        classes = [[], []]
        l = len(dataset)
        for xi, label in zip(dataset, labels):
            classes[int(label)].append(xi)

        scores = [self._scores(data_point, label, classes) for data_point, label in zip(dataset, labels)]

        return map(self._feature_correlation, zip(*scores))
