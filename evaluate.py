from decision_tree import Decision_tree
from decision_tree import decision_tree_learning
import numpy as np
import sys
import matplotlib.pyplot as plot


np.random.seed(12306)

LABEL_NUM = 4
NUM_FOLD = 10

# calculates the confusion matrix
def get_confusion_matrix(test_data, decision_tree):
    confusion_matrix = [[0 for _ in range (LABEL_NUM)] for _ in range (LABEL_NUM)]

    #matrix [x][y] means the model label x and the actual prediction is y
    for data in test_data:
        features = data[:-1]
        label = data[-1]
        confusion_matrix[int(label)-1][int(decision_tree.predict(data)) - 1] += 1

    return confusion_matrix

# calculates recall
def recall(confusion_matrix, class_label):
    matrix_index = class_label -1 
    return confusion_matrix[matrix_index][matrix_index] / sum(confusion_matrix[matrix_index])

# calculates precision
def precision(confusion_matrix, class_label):
    matrix_index = class_label -1 
    return confusion_matrix[matrix_index][matrix_index] / sum([row[matrix_index] for row in confusion_matrix])

# calculates f1
def f1(confusion_matrix, class_label):
    p = precision(confusion_matrix, class_label)
    r = recall(confusion_matrix, class_label)
    if p+r == 0:
        return 0
    return (2 * p * r) / (p + r)

# calculates accuracy
def accuracy(confusion_matrix):
    correct = 0
    total = 0
    for i in range (LABEL_NUM):
        for j in range (LABEL_NUM):
            total += confusion_matrix[i][j]
            if i == j:
                correct += confusion_matrix[i][j]
    return correct / total

# outputs performance of a model given a dataset, and returns confusion matrix for display
def experiment(dataset, prune = False):
    np.random.shuffle(dataset)

    partitions = np.array_split(dataset, NUM_FOLD)

    recalls = [0 for i in range (LABEL_NUM)]
    precisions = [0 for i in range (LABEL_NUM)]
    f1s = [0 for i in range (LABEL_NUM)]
    average_accuracy = 0
    average_confusion_matrix = [[0 for i in range (LABEL_NUM)] for _ in range (LABEL_NUM)]

    for i in range (NUM_FOLD):
        # divides dataset into partitions for training, testing and validation
        test_set = partitions[i]
        if prune:
            if i == 0:
                training_set = np.concatenate(partitions[1: - 1])
            else:
                training_set = np.concatenate(partitions[:i - 1] + partitions[i + 1:])
            validation_set = partitions[i - 1]
            decision_tree, _ = decision_tree_learning(training_set, 1)
            decision_tree = decision_tree.prune(validation_set)
            depth = decision_tree.depth()
        else:
            training_set = np.concatenate(partitions[:i] + partitions[i + 1:])
            decision_tree, depth = decision_tree_learning(training_set, 1)
        confusion_matrix = get_confusion_matrix(test_set, decision_tree)

        # calculates performance
        for j in range (LABEL_NUM):
            precisions[j] += precision(confusion_matrix, j) / NUM_FOLD
            recalls[j] += recall(confusion_matrix, j) / NUM_FOLD
            f1s[j] += f1(confusion_matrix, j) / NUM_FOLD
        average_accuracy += accuracy(confusion_matrix) / NUM_FOLD

        # generates confusion matrix
        for i in range (LABEL_NUM):
            for j in range (LABEL_NUM):
                average_confusion_matrix[i][j] += confusion_matrix[i][j] / NUM_FOLD

    print ("tree depth :", depth)
    print ("confusion matrix :", average_confusion_matrix)

    print ("precision for the four classes :", precisions)
    print ("recall for the four classes :", recalls)
    print ("f1 for the four classes :", f1s)
    print ("accuracy :", average_accuracy)

    return confusion_matrix

# displays the confusion matrix
def display_confusion_matrix(fig,confusion_matrix, subplot_num, subplot_title):

    ax = fig.add_subplot(220 + subplot_num)

    ax.matshow(confusion_matrix, cmap='seismic')
    ax.set_xticklabels([0, 1, 2, 3, 4])
    ax.set_yticklabels([0, 1, 2, 3, 4])
    ax.set_title(subplot_title)
    ax.set_xlabel('predicted label')
    ax.set_ylabel('actual label')
    for (i, j), z in np.ndenumerate(np.array(confusion_matrix)):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    

# main function
def main(args):
    if args == []:
        clean_data = np.loadtxt("clean_dataset.txt")
        noisy_data = np.loadtxt("noisy_dataset.txt")

        fig = plot.figure()
        fig.suptitle('Confusion matrices for the experiments', fontsize = 16)

        print ("experiment on clean dataset")
        display_confusion_matrix(fig,experiment(clean_data), 1, "clean data without prune")
        print ("\n\n====================================================================\n\n")
        print ("experiment on noisy dataset")
        display_confusion_matrix(fig,experiment(noisy_data), 2, "noisy data without prune")

        print ("\n\n====================================================================\n\n")
        print ("experiment on clean dataset after pruning")
        display_confusion_matrix(fig,experiment(clean_data, prune = True), 3, "clean data with prune")

        print ("\n\n====================================================================\n\n")
        print ("experiment on noisy dataset after pruning")
        display_confusion_matrix(fig,experiment(noisy_data, prune = True), 4, "noisy data with prune")

        plot.show()

    else:
        custom_dataset = np.loadtxt(args[0])
        print ("experiment on custom dataset without pruning")
        experiment(custom_dataset)
        print ("\n\n====================================================================\n\n")
        print ("experiment on custom dataset with pruning")
        experiment(custom_dataset, prune = True)
if __name__ == '__main__':
    main(sys.argv[1:])