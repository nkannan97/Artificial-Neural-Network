import numpy as np
import matplotlib.pyplot as plt
from NN import *
from cross_validation import *
from NeuralNet import *
from operator import itemgetter


def k_fold_cross_validation(featured_data,num_folds,num_epochs):

    validation = cross_val(featured_data,num_folds)
    features_data = validation.split_data(validation.data, label_range)
    results = []
    track_test_index = 0

    for fold_index in range(num_folds+1):

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

    #acc = calc_and_print(results,'eval')
    return results

number_epochs = [25,50,75,100]
number_folds = [5,10,15,20,25]
acc_lst_for_e = []
acc_lst_for_nf = []

#train_file, num_folds, learning_rate, num_epochs = reading_input_arguments()
featured_data, data, metadata, feature_range = reading_data('sonar.arff')
label_range = feature_range[-1]
num_inputs = len(metadata.types()) - 1
num_outputs = 1

# for num_e in number_epochs:
#     acc_lst_for_e.append(k_fold_cross_validation(featured_data,10,num_e))
#
#
# print(acc_lst_for_e)
# plt.plot([t for t in number_epochs], [x*100.0 for x in acc_lst_for_e])
# plt.xlabel('num_epochs')
# plt.ylabel('Accuracy')
# plt.title('Sonar')
# plt.ylim((0,90))
# plt.xlim((0,125))
# plt.show()

# for num_nf in number_folds:
#     acc_lst_for_nf.append(k_fold_cross_validation(featured_data,num_nf,50))
# print(acc_lst_for_nf)
# plt.plot([t for t in number_folds], [x*100.0 for x in acc_lst_for_nf])
# plt.xlabel('num_folds')
# plt.ylabel('Accuracy')
# plt.title('Sonar')
# plt.ylim((0,90))
# plt.xlim((0,30))
# plt.show()

# # consider rock as pos
num_pos = 0
# consider mine as neg
num_neg = 0



new_thresh = []

def ROC_threshold(featured_data, acc, num_neg, num_pos,TP,FP,last_TP):
    thresholds = []
    for i in range(len(acc)):

        if i > 0 and acc[i][4] != acc[i - 1][4] and acc[i][3] == label_range[1] and TP > last_TP:
            FPR = 1.0 * FP / num_neg
            TPR = 1.0 * TP / num_pos
            thresholds.append((FPR, TPR))
            last_TP = TP

        if acc[i][3] == label_range[0]:
            TP += 1
        else:
            FP += 1
    FPR = FP/num_neg
    TPR = TP/num_pos
    thresholds.append((FPR,TPR))
    return thresholds


acc = k_fold_cross_validation(featured_data,10,50)
acc.sort(key=lambda x: x[4])
print(acc)
for instance in featured_data:
    if str(instance[-1], 'utf-8') == label_range[0]:
        num_pos += 1
    else:
        num_neg += 1
new_thresh = ROC_threshold(featured_data,acc,num_neg, num_pos,0,0,0)
print(new_thresh)

plt.plot([t[0] for t in new_thresh], [x[1] for x in new_thresh])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
#plt.ylim((0,90))
#plt.xlim((0,125))
plt.show()
