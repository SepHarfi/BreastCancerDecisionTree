import copy
import numpy as np
import math
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Class for storing Decision Tree as a binary-tree
        Inputs:
        - feature: Name of the feature based on which this node is split
        - threshold: The threshold used for splitting this subtree
        - left: left Child of this node
        - right: Right child of this node
        - value: Predicted value for this node (if it is a leaf node)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        if self.value is not None:
            return True

        return False


class DecisionTree:
    def __init__(self, max_depth=1e9, min_samples_split=2):
        """
        Class for implementing Decision Tree
        Attributes:
        - max_depth: int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until
            all leaves contain less than min_samples_split samples.
        - min_num_samples: int
            The minimum number of samples required to split an internal node
        - root: Node
            Root node of the tree; set after calling fit.
        """
        self.max_depth = max_depth
        self.depth = 0
        self.min_samples_split = min_samples_split
        self.root = None

    def is_splitting_finished(self, depth, num_class_labels, num_samples):
        """
        Criteria for continuing or finishing splitting a node
        Inputs:
        - depth: depth of the tree so far
        - num_class_labels: number of unique class labels in the node
        - num_samples: number of samples in the node
        :return: bool
        """
        if num_class_labels == 1 or num_samples < self.min_samples_split or depth > self.max_depth:
            return True

        return False

    def split(self, X, y, feature, threshold):
        """
        Splitting X and y based on value of feature with respect to threshold;
        i.e., if x_i[feature] <= threshold, x_i and y_i belong to X_left and y_left.
        Inputs:
        - X: Array of shape (N, D) (number of samples and number of features respectively), samples
        - y: Array of shape (N,), labels
        - feature: Name of the feature based on which split is done
        - threshold: Threshold of splitting
        :return: X_left, X_right, y_left, y_right
        """
        X_left = X[X[feature] <= threshold]
        X_right = X[X[feature] > threshold]
        y_left = y[X[feature] <= threshold]
        y_right = y[X[feature] > threshold]

        return X_left, X_right, y_left, y_right

    def entropy(self, y):
        """
        Computing entropy of input vector
        - y: Array of shape (N,), labels
        :return: entropy of y
        """
        p0 = len(y[y == 0]) / len(y)
        p1 = len(y[y == 1]) / len(y)
        if p0 == 0 or p1 == 0:
            return 1e9
        return - p0 * math.log2(p0) - p1 * math.log2(p1)

    def information_gain(self, X, y, feature, threshold):
        """
        Returns information gain of splitting data with feature and threshold.
        Hint! use entropy of y, y_left and y_right.
        """
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            return -1
        h_y = self.entropy(y)
        h_yl = self.entropy(y_left)
        h_yr = self.entropy(y_right)
        p_r = len(y_right) / len(y)
        p_l = len(y_left) / len(y)
        h_y_cond = p_r * h_yr - p_l * h_yl

        return h_y - h_y_cond

    def best_split(self, X, y):
        """
        Used for finding best feature and best threshold for splitting
        Inputs:
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        :return:
        """
        features = list(X.columns)  # todo: You'd better use a random permutation of features 0 to D-1
        best_feature = features[0]
        best_feature_ig = 0
        best_feature_thresh = 0
        for feature in features:
            # todo: use unique values in this feature as candidates for best threshold
            thresholds = list(np.arange(X[feature].min(), X[feature].max(), 40))
            best_threshold = 0
            best_thresh_ig = 0
            for threshold in thresholds:
                new_ig = self.information_gain(X, y, feature, threshold)
                if new_ig > best_thresh_ig:
                    best_thresh_ig = new_ig
                    best_threshold = threshold
            if best_feature_ig < best_thresh_ig:
                best_feature = feature
                best_feature_ig = best_thresh_ig
                best_feature_thresh = best_threshold

        return best_feature, best_feature_thresh

    def build_tree(self, X, y, depth=0):
        """
        Recursive function for building Decision Tree.
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        - depth: depth of tree so far
        :return: root node of subtree
        """
        self.depth = max(self.depth, depth)
        best_feature, best_feature_thresh = self.best_split(X, y)
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_feature_thresh)
        if self.is_splitting_finished(depth, len(set(y_left)), len(X_left)):
            node_left = Node(value=set(y_left).pop())
        else:
            node_left = self.build_tree(X_left, y_left, depth + 1)

        if self.is_splitting_finished(depth, len(set(y_right)), len(X_right)):
            node_right = Node(value=set(y_right).pop())
        else:
            node_right = self.build_tree(X_right, y_right, depth + 1)

        return Node(best_feature, best_feature_thresh, node_left, node_right, value=None)



    def fit(self, X, y):
        """
        Builds Decision Tree and sets root node
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        """

        return self.build_tree(X, y, 0)


    def predict(self, X, root):
        """
        Returns predicted labels for samples in X.
        :param X: Array of shape (N, D), samples
        :return: predicted labels
        """
        target = []
        for row_i in range(len(X)):
            row = X.iloc[row_i]
            node = copy.deepcopy(root)
            while True:
                if node.is_leaf():
                    target.append(node.value)
                    break
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        data_frame = pd.DataFrame({"target": target})
        return data_frame


# import data

# Split your data to train and validation sets

# Tune your hyper-parameters using validation set

# Train your model with hyper-parameters that works best on validation set

# Predict test set's labels

test = pd.read_csv("test.csv")
training_data = pd.read_csv("breast_cancer.csv")
y = training_data['target'][:int(0.7 * len(training_data))]
y2 = (training_data['target'][int(0.7 * len(training_data)):].reset_index()).drop(columns=['index'])

training_data = training_data.drop(columns=["target"])
validation_set = (training_data[int(0.7
                                    * len(training_data)):].reset_index()).drop(columns=['index'])
training_data = training_data[:int(0.7 * len(training_data))]

tree = DecisionTree()
root = tree.fit(training_data, y)

'''max_depth = tree.depth
min_samples_split = 2

best_samples_split = 2
best_max_depth = tree.depth
best_a = 0
for i in range(max_depth, max_depth - 5, -1):
    for j in range(min_samples_split, min_samples_split + 6, 2):
        tree = DecisionTree(i, j)
        root = tree.fit(training_data, y)

        y3 = tree.predict(validation_set, root)
        a = len(y3[y3 == y2]) / len(y3)

        if a > best_a:
            best_samples_split = j
            best_max_depth = i
            best_a = a

tree = DecisionTree(best_max_depth, best_samples_split)
root = tree.fit(training_data, y)'''
data_frame = tree.predict(test, root)
data_frame.to_csv("output.csv", index=False)