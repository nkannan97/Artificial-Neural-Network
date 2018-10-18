import numpy as np

class cross_val():

    def __init__(self, training_data, num_folds):

        self.data = training_data
        self.num_folds = num_folds
        self.folds = {}
        self.indexfolds = {}

    def split_data(self, data, label_range):

        label_0 = []
        label_1 = []
        for i in range(len(data)):

            if str(data[i][-1], 'utf-8') == label_range[0]:
                label_0.append(i)
            else:
                label_1.append(i)

        for indices in range(self.num_folds):
            self.indexfolds[indices] = []
            self.folds[indices] = []

        lst = [label_0, label_1]

        for index in range(len(label_range)):
            np.random.shuffle(lst[index])

        ratio_0 = len(label_0)/len(data)
        ratio_1 =  1 - ratio_0

        instances_per_fold = np.round(len(data)/self.num_folds)

        label_0_instances = np.round(ratio_0*instances_per_fold)
        label_1_instances = instances_per_fold - label_0_instances
        count_0 = 0
        count_1 = 0

        common_lst =[]

        for j in range(self.num_folds):
            while count_0 < label_0_instances and label_0 != []:
                common_lst.append(label_0.pop())
                count_0 += 1
            while count_1 < label_1_instances and label_1 != []:
                common_lst.append(label_1.pop())
                count_1 += 1
            count_0 = 0
            count_1 = 0

            self.folds[j] = common_lst
            self.indexfolds[j] = self.folds[j]
            common_lst = []

        if label_0 != [] or label_1 != []:
            self.folds[self.num_folds] = list(set(label_0) | set(label_1))



    def split_train_test(self, fold_index,label_range):

        get_fold_indices = list(self.folds[fold_index])
        get_fold_indices.sort()
        training_data= []
        testing_data = []

        for indices in range(len(self.data)):

            if indices in get_fold_indices:
                testing_data.append(self.data[indices])
            else:
                training_data.append(self.data[indices])

        np.random.shuffle(training_data)
        #print('indices: ', get_fold_indices)
        #print('testing_data: ', testing_data[0])
        return training_data, testing_data, get_fold_indices



