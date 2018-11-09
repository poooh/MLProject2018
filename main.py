import sys
from lib.data_wrangler import DataWrangler
from lib.evaluator import Evaluator
from learner import NaiveBayesClassifier


def get_file_paths_stdin():
    l = len(sys.argv)
    if l != 3:
        err_msg = "Error. Signature to run file is- python main.py "
        err_msg = err_msg + "<path_to_dataset_file> <path_to_training_file>"
        print(err_msg)
        sys.exit(0)

    return sys.argv[1:]


def get_data(path_to_dataset_file, path_to_training_file):
    dataset = DataWrangler.read_from_file(path_to_dataset_file, conversion=float)
    train_labels_indices = DataWrangler.read_from_file(path_to_training_file)

    return DataWrangler.separate_train_test_data(dataset, train_labels_indices)


if __name__ == "__main__":
    path_to_dataset_file, path_to_training_file = get_file_paths_stdin()

    train_features, test_features, test_indices, train_labels = get_data(path_to_dataset_file, path_to_training_file)

    model = NaiveBayesClassifier(classes=2)
    model.fit(train_features, train_labels)

    predicted_labels = model.predict(test_features)
    if predicted_labels:
        for i, j in zip(predicted_labels, test_indices):
            print(i, j)
