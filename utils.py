from math import sqrt



class NaiveBayesHelper:
    @classmethod
    def _get_shape(cls, X):
        """FUNCTION TO GET SHAPE OF FEATURES DATA"""

        return len(X), len(X[0])

    @classmethod
    def get_mean_std_dev(cls, X, y, classes):
        """FUNCTION TO GET CLASS MEAN AND STANDARD DEVIATION"""

        n_samples, n_features = cls._get_shape(X)

        class_means, class_variance = [], []
        for i in range(classes):
            class_means.append([1.0] * n_features)
            class_variance.append([0.0] * n_features)

        class_count = [0] * classes
        for i in range(n_samples):
            for j in range(n_features):
                class_means[y[i]][j] += X[i][j]
            class_count[y[i]] += 1

        for i in range(classes):
            for j in range(n_features):
                class_means[i][j] /= float(class_count[i])

        for i in range(n_samples):
            for j in range(n_features):
                class_variance[y[i]][j] += (X[i][j] - class_means[y[i]][j]) ** 2
        class_std_dev = []
        for i in range(classes):
            std_dev = []
            for j in range(n_features):
                class_variance[i][j] /= float(class_count[i])
                std_dev.append(sqrt(class_variance[i][j]))
            class_std_dev.append(std_dev)

        return class_means, class_std_dev

    @staticmethod
    def get_normalized_vector_distance(vector_a, vector_b, normalization):
        """FUNCTION TO CALCULATE NORMALIZED DISTANCE BETWEEN TWO VECTORS"""
        distance = 0.0
        for i, j, k in zip(vector_a, vector_b, normalization):
            distance += ((i - j) / k) ** 2

        return sqrt(distance)