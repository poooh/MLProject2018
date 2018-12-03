from time import time
from random import random, shuffle
import sys, os

from sklearn.svm import SVC

from lib.data_wrangler import DataWrangler
from lib.utility import Evaluation
from feature_selection.feature_selector import SelectionCombinator

print_prefix = "Time taken to complete"


def get_data():
    if len(sys.argv) < 4:
        print("Error running file. Signature to run the file is: -")
        print("python main.py <TRAIN_DATA_FILE_PATH> <TEST_DATA_FILE_PATH> <TRAINLABEL_FILE_PATH>")
        sys.exit(0)

    train_data = DataWrangler.read_from_file(sys.argv[1], conversion=float)
    test_data = DataWrangler.read_from_file(sys.argv[2], conversion=float)
    train_labels = DataWrangler.read_from_file(sys.argv[3], conversion=int)
    zipped_train = list(zip(train_data, train_labels))
    shuffle(zipped_train)
    train_data, train_labels = zip(*zipped_train)
    training_set = []
    validation_set = []
    training_labels = []
    validation_labels = []
    for xi, label in zip(train_data, train_labels):
        if random() > 0.1:
            training_set.append(xi)
            training_labels.append(label)
        else:
            validation_set.append(xi)
            validation_labels.append(label)

    return training_set, validation_set, test_data, list(list(zip(*training_labels))[0]), \
           list(list(zip(*validation_labels))[0])


if __name__ == "__main__":
    def reduce_dimension(dataset, selected_features):
        return [[xi[i] for i in selected_features] for xi in dataset]


    super_start = time()
    train_data, validation_data, test_data, train_labels, validation_labels = get_data()
    print("{} dataset load = {} seconds".format(print_prefix, time() - super_start))

    start = time()
    combination = (('snr', 1000), ('pearson', 200), ('mi', 15))
    combinator = SelectionCombinator(combination)
    reduced_train_data, reduced_validation_data, reduced_test_data = combinator.get_reduced_data(train_data,
                                                                                                 validation_data,
                                                                                                 test_data,
                                                                                                 train_labels)
    del train_data
    del validation_data
    del test_data
    print("{} dataset reduction = {} seconds".format(print_prefix, time() - start))

    start = time()
    model = SVC(kernel="linear", C=3.0, max_iter=1000000)
    predictions = model.fit(reduced_train_data, train_labels).predict(reduced_validation_data)
    accuracy = Evaluation.get_accuracy(predictions, validation_labels)
    print("{} model learning and predicting = {} seconds".format(print_prefix, time() - start))
    print("Got {} accuracy using linear SVM for 15 dims".format(accuracy))

    start = time()
    train_data = reduced_train_data + reduced_validation_data
    del reduced_train_data
    del reduced_validation_data
    train_labels = train_labels + validation_labels
    model = SVC(kernel="linear", C=6.0)
    predictions = model.fit(train_data, train_labels).predict(reduced_test_data)

    prediction_output_file = os.path.join(os.path.dirname(__file__), 'final_predictions.txt')
    DataWrangler.write_to_file(prediction_output_file, predictions)
    print("{} test prediction writing = {} seconds".format(print_prefix, time() - start))

    print("{} full processing = {} seconds".format(print_prefix, time() - super_start))
