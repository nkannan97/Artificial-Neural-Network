import numpy as np
from scipy.io import arff
from NN import Neural_Network
from cross_validation import cross_val
import sys


def train_neural_net(perceptron, train_data, test_data, label_range, num_epochs, learning_rate):
    perceptron.set_learning_rate(learning_rate)
    for epoch in range(num_epochs):
        for instance in train_data:
            feature_array, label = convert_to_vector(instance, None, label_range, 'train')
            feature_instance = feature_array.reshape((len(feature_array[0]), 1))
            NN_output = perceptron.feed_forward(feature_instance)
            perceptron.backprop(feature_instance, label, NN_output)


def convert_to_vector(training_instance, testing_instance, label_range, check):
    feature_array = np.zeros((1, len(training_instance) - 1))
    if check == 'train':
        for i in range(len(training_instance) - 1):
            feature_array[0][i] = training_instance[i]

        if str(training_instance[i + 1], 'utf-8') == label_range[0]:
            label = 0
        else:
            label = 1

    else:
        for i in range(len(testing_instance) - 1):
            feature_array[0][i] = testing_instance[i]

        if str(testing_instance[i + 1], 'utf-8') == label_range[0]:
            label = 0
        else:
            label = 1

    return feature_array, label


def calc_and_print(results, type):
    correct_prediction = 0
    results.sort(key=lambda x: x[0])
    for map_size in range(len(results)):
        if type == 'print':
            print('{} {} {} {:0.6f}'.format(results[map_size][1], results[map_size][2], results[map_size][3],
                                            results[map_size][4]))

        if results[map_size][2] == results[map_size][3]:
            correct_prediction += 1

    if type == 'eval':
        acc = 1.0 * correct_prediction / len(results)
        return acc


def reading_input_arguments():
    if len(sys.argv)!= 5:
        sys.exit('not enough input arguments')
    train_file = str(sys.argv[1])
    num_folds = int(str(sys.argv[2]))
    learning_rate = float(str(sys.argv[3]))
    num_epochs = int(str(sys.argv[4]))

    return train_file, num_folds, learning_rate, num_epochs


def reading_data(training_file):
    diff_feature_values = []
    features, metadata = arff.loadarff(training_file)
    for feature_names in metadata.names():
        diff_feature_values.append(metadata[feature_names][1])
    return features, features, metadata, diff_feature_values


