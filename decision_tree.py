import numpy as np

class Decision_tree:
    
    # initialise decision tree
    def __init__(self,feature_index, threshold, left_sub_tree, right_sub_tree, label = -1, majority_label = -1):
        self.__feature_index = feature_index
        self.__threshold = threshold
        self.__left_sub_tree = left_sub_tree
        self.__right_sub_tree = right_sub_tree
        self.__label = label
        self.__majority_label = majority_label

    # return whether not the parameter node is a leaf
    def is_leaf(self):
        return self.__feature_index == -1

    # uses the model to make predictions
    def predict(self, data):
        if self.__label != -1:
            return self.__label

        if data[self.__feature_index] < self.__threshold:
            return self.__left_sub_tree.predict(data)
        
        return self.__right_sub_tree.predict(data)

    def visualize(self):
        pass

    # returns the depth of the tree
    def depth(self):
        if self.is_leaf():
            return 1
        return 1 + max(self.__left_sub_tree.depth(), self.__right_sub_tree.depth())

    # recursively prunes the decision tree
    def prune(self, validation_set):
        if self.is_leaf() or len(validation_set) == 0:
            return self

        # splits the validation set into appropriate subtree
        l_element_index = []
        r_element_index = []
        for i in range(len(validation_set)):
            if validation_set[i][self.__feature_index] < self.__threshold:
                l_element_index.append(i)
            else:
                r_element_index.append(i)

        l_dataset = validation_set[l_element_index]
        r_dataset = validation_set[r_element_index]

        self.__left_sub_tree = self.__left_sub_tree.prune(l_dataset)
        self.__right_sub_tree = self.__right_sub_tree.prune(r_dataset)

        # runs validation set on unpruned tree
        correct_unpruned = 0
        for data in validation_set:
            features = data[:-1]
            label = data[-1]
            if self.predict(features) == label:
                correct_unpruned += 1

        # runs validation set on pruned tree
        pruned_tree = Decision_tree(-1, -1, None, None, 
                label = self.__majority_label)
        correct_pruned = 0
        for data in validation_set:
            features = data[:-1]
            label = data[-1]
            if pruned_tree.predict(features) == label:
                correct_pruned += 1       

        if correct_pruned >= correct_unpruned:
            return pruned_tree
        return self

# returns the entropy of a dataset
def entropy(dataset):
    labels = dataset[:,-1]
    value,counts = np.unique(labels, return_counts = True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(2)).sum()

# finds the optimal feature to split on, and splits the dataset
def split(dataset):
    data = dataset[:,:-1]
    min_remainder_entropy = float('inf')
    l_set, r_set, index, threshold = None, None, -1, -1

    for i in range(len(dataset[0])-1):
        # sort the dataset and split in half
        dataset = dataset[dataset[:,i].argsort()]
        l_dataset, r_dataset = np.array_split(dataset, 2)

        l_entropy = entropy(l_dataset)
        r_entorpy = entropy(r_dataset)

        # does this feature result in a new lowest remainder
        if l_entropy * 1/2 + r_entorpy * 1/2 < min_remainder_entropy:
            min_remainder_entropy = l_entropy * 1/2 + r_entorpy * 1/2
            l_set, r_set, index, threshold = l_dataset, r_dataset, i, (l_dataset[-1][i] + r_dataset[0][i]) / 2

    assert(l_set is not None)
    assert(r_set is not None)

    return index, threshold, l_set, r_set

# builds the decision tree using the given training set
def decision_tree_learning(dataset, depth):
    labels = dataset[:, -1]    
    if np.all(labels == labels[0]):
        return Decision_tree(-1, -1, None, None, labels[0]), depth

    values, counts = np.unique(labels, return_counts = True)
    majority_label = values[np.argmax(counts)]

    feature_index, threshold, l_dataset, r_dataset = split(dataset)

    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

    return (Decision_tree(feature_index,threshold, l_branch, r_branch, majority_label = majority_label), max(l_depth, r_depth))


