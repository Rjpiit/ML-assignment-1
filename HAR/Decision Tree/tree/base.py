"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index


np.random.seed(42)


class node:
    def __init__(self, val=None, depth=None, col_used=None):

        self.val = val  # stores the value of the node, mean/ mode of the output
        self.depth = depth  # depth of the node in the decision tree
        self.child_ = {}  # dictionary containing the children of the node
        self.col_used = col_used  # the name of the column used by the node to split further
        self.split_val = None  # the mean value used to split the table in case of continous input
        self.prob = None
        # in order to determine with the node is a leaf node
        if(col_used == None):
            self.is_leaf = True
        else:
            self.is_leaf = False
    # the function to return the predicted output for a given input dataframe

    def ret_val(self, X, max_depth=np.inf):
        # the node is a leaf node or the max_depth has been reached
        if(self.is_leaf or self.depth >= max_depth):
            return(self.val)  # the value of the node is returned

        else:
            if(self.split_val == None):  # in case the node takes discrete values
                # checking if the attribute value is present in the children nodes
                if(X[self.col_used] in self.child_):
                    # recursively calling ret_val on the corresponding node
                    return self.child_[X[self.col_used]].ret_val(X, max_depth=max_depth)
                else:
                    # in case the attribute value is not present
                    prob_max = 0
                    child = None
                    for xi in self.child_:
                        if self.child_[xi].prob > prob_max:
                            prob_max = self.child_[xi].prob
                            child = self.child_[xi]
                    # recursively pass the function to the node with the highest probability
                    return(child.get_node_val(X.drop(self.col_used), max_depth=max_depth))
            else:  # in case the node takes continous values
                if(X[self.col_used] <= self.split_val):
                    # comparing the checking the corresponding attribute value and recursively passing to the corresponding node
                    return(self.child_["left"].ret_val(X, max_depth=max_depth))
                else:
                    return(self.child_["right"].ret_val(X, max_depth=max_depth))

    # A function used to print the whole tree which considers the cases of node being leaf or not
    def _print_(self, gap=0, sta=None):
        if self.is_leaf:  # Prints all the leaf nodes
            if self.val.dtype.name == "str":
                if(sta == None):
                    print("|   "*gap +
                          "|--- val = {} Depth = {}".format(self.val, self.depth))
                else:
                    print("|   "*gap + sta +
                          "|--- val = {} Depth = {}".format(self.val, self.depth))
            else:
                if(sta == None):
                    print("|   "*gap +
                          "|--- val = {:.2f} Depth = {}".format(self.val, self.depth))
                else:
                    print("|   "*gap + sta +
                          "|--- val = {:.2f} Depth = {}".format(self.val, self.depth))
        else:
            Hi = False
            for xi in self.child_:
                if self.child_[xi].prob != None:
                    Hi = True
            if(Hi):  # for non-leaf node
                if sta == None:
                    print("|   "*gap, end="")
                else:
                    print("|   "*gap + sta, end="")
                count = 0
                for xi in self.child_:  # for discrete input, to print the checking of equality to the attribute value
                    if count == 0:
                        print("| ?(X({}) = {}):".format(self.col_used, xi))
                        count = 9287
                    else:
                        print("|   "*gap +
                              "| ?(X({}) = {}):".format(self.col_used, xi))

                    self.child_[xi]._print_(gap + 1)
            else:  # for continuous input
                # checking the sign of child
                if(sta == None):
                    sta = ""
                sign = ">"
                print("|   "*gap + sta + "| ?(X({}) {} {:.2f}):".format(self.col_used,
                                                                        sign, self.split_val))
                sta = None
                for xi in self.child_:
                    if xi == "left":
                        sta = "N: "
                    else:
                        sta = "Y: "
                    # recursively pasing to the next node with an increment in the gap
                    self.child_[xi]._print_(gap + 1, sta)


class DecisionTree:
    # criterion won't be used for regression
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=20):
        # to record the criterion measure the randomness of the data
        self.criterion = criterion
        self.max_depth = max_depth  # max_depth that can be reached
        self.root = None           # to store the root of the decision tree
        self.Y_name = None         # name of the output column
        self.y_type = None         # data type of output
        self.size_X = None         # row size of the input dataframe

    # function to build the tree
    def rec_build(self, X, Y, cur_root=None, depth=0):

        # base case : output has only one value
        if(len(Y.unique()) == 1):
            # node is added with value equal to the single output value
            return node(val=Y[0], depth=depth)

        # ensuring that
        # 1. there is atleast one column with more than one unique value
        # 2. The depth limit hasnt been reached yet
        # 3. the input dataframe has atleast one entry

        elif((max(list(X.nunique()))) > 1 and X.size > 0 and depth < self.max_depth):
            # Finding the column with the maximum information gain
            info_max = -np.inf
            store_tup = None  # to store the value of the output tuple
            col_used = None  # the index of the column used in the split
            # iterating throught the columns inorder to find the attribute that gives that maximum info_gain
            for column in X:
                attr = pd.Series(X[column])
                tup_out = information_gain(Y, attr, self.criterion)
                info_gain = tup_out[1]
                if(info_gain > info_max):
                    info_max = info_gain
                    store_tup = tup_out
                    col_used = column
            # Now we have located the column that has to be used for the split
            node_new = node(col_used=col_used)
            # to store the best split value in case of continous attribute and None value otherwise
            split = store_tup[0]
            attr = pd.Series(X[col_used])
            dict1 = {1: Y, 2: attr}
            dict1 = pd.concat(dict1, axis=1)
            if(split == False):  # this implies that the input column is discrete

                uni_attr = attr.unique()
                for xi in uni_attr:
                    # creating a dataframe with the corresponding attribute value for the attribute column
                    df1 = X[X[col_used] == xi].reset_index(drop=True)
                    # removing the column used for splitting the data
                    df1.drop(col_used, axis=1)
                    dict_ini_2 = dict1[dict1[2] == xi]
                    y_ = dict_ini_2.iloc[:, 0].reset_index(drop=True)
                    # building a subtree with sub table obtained and passing the corresponding output
                    node_new.child_[xi] = self.rec_build(
                        df1, y_, node_new, depth=depth+1)
                    # setting the prob of getting each child
                    node_new.child_[xi].prob = len(df1)/self.size_X
            else:  # in case the input column is continous
                # We dont have to drop any columns in this case
                node_new.split_val = split
                # print(node_new.split_val)
                # spliting the dataframe and the outputs and passing them recursively to make the corresponding subtrees
                X_left = X[X[col_used] <= split].reset_index(drop=True)
                X_right = X[X[col_used] > split].reset_index(drop=True)

                y_left = (dict1[dict1[2] <= split]).iloc[:,
                                                         0].reset_index(drop=True)
                y_right = (dict1[dict1[2] > split]).iloc[:,
                                                         0].reset_index(drop=True)

                node_new.child_["right"] = self.rec_build(
                    X_right, y_right, node_new, depth=depth+1)
                node_new.child_["left"] = self.rec_build(
                    X_left, y_left, node_new, depth=depth+1)
            # setting the value of the newly created node
            if Y.dtype.name == "category":  # in case of discrete output
                node_new.val = Y.mode()[0]
            else:  # in case of continous output
                node_new.val = Y.mean()
            # setting the depth of the current node
            node_new.depth = depth

            return node_new
        else:  # in all the cases left(max-depth, end of dataset, no district inputs) mean/mode has to be returned
            if Y.dtype.name == "category":  # in case of discrete input
                return node(val=Y.mode()[0], depth=depth)
            else:  # in case of continous input
                return node(val=Y.mean(), depth=depth)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # recording the output type and intialising the tree formation
        self.Y_name = Y.name  # assigning the name of the output column
        self.size_X = len(X)
        # Creating the tree and storing the root node
        self.root = self.rec_build(X=X, Y=Y)
        self.y_type = Y.dtype
        self.root.prob = 1  # probability of the root node is always 1

    def predict(self, X: pd.DataFrame, max_depth=np.inf):
        """
        Funtion to run the decision tree on test inputs
        """
        Y = []  # initialising a list to store the predicted output values
        for xi in X.index:
            # passing the corresponding row to ret_val to get the output
            Y.append(self.root.ret_val(X.loc[xi], max_depth=max_depth))
        # return the output stored in the form a Series
        return(pd.Series(Y, name=self.Y_name).astype(self.y_type))

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root._print_()
