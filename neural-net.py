import numpy as np
from NeuralNet import *
from cross_validation import *
from NN import *

train_file, num_folds, learning_rate, num_epochs = reading_input_arguments()
featured_data, data, metadata, feature_range = reading_data(train_file)
label_range = feature_range[-1]
num_inputs = len(metadata.types()) - 1
num_outputs = 1
validation = cross_val(featured_data, num_folds)
validation.split_data(validation.data, label_range)
results = []
track_test_index = 0

for fold_index in range(num_folds + 1):

    training_data, testing_data, test_indices = validation.split_train_test(fold_index, label_range)
    network = Neural_Network(num_inputs, num_outputs)
    network.initialization_weights_biases()
    train_neural_net(network, training_data, testing_data, label_range, num_epochs, 0.1)

    for test_instance in testing_data:
        actual_label = test_instance[-1]
        test_set, numeric_label = convert_to_vector(test_instance, test_instance, label_range, 'test')
        test_instance_reshape = test_set.reshape((len(test_set[0]), 1))
        confidence = network.feed_forward(test_instance_reshape)

        if confidence <= 0.5:
            predicted_label = label_range[0]
        else:
            predicted_label = label_range[1]

        results.append([test_indices[track_test_index], fold_index, predicted_label, str(actual_label, 'utf-8'),
                        np.round(confidence[0][0], 6)])
        track_test_index += 1

    track_test_index = 0

calc_and_print(results, 'print')
