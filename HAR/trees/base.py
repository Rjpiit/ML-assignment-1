"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

np.random.seed(42)

def all_elements_equal(series):
    return series.nunique() == 1

@dataclass
# Node1 is for building the decision tree for (Discrete input, Real output) to avoid confusion while merging others' code
class Node1:
    def __init__(self, X: pd.DataFrame, y: pd.Series, max_depth = 5, name=None, depth=0, crit="information_gain"):
        self.name = name
        self.depth = depth
        feature = opt_split_attribute(X,y)
        self.feature = feature
        self.value = None
        self.child = []
        if max_depth == 1 or all_elements_equal(y):
            avg = 0
            for i in y:
                avg += i
            avg = avg/len(y)
            self.value = avg
            return None
        else:
            for each in X[feature].unique():
                X_new = X[X[feature] == each]
                y_new = y[X[feature] == each]
                X_new = X_new.drop(feature, axis=1)
                self.child.append(Node1(X_new,y_new,max_depth-1,each, depth+1))
    
    def _print_(self, gap=0):
        if self.value is not None:  # Prints all the leaf nodes
            print("|   "*gap + "|--- val = {:.2f} Depth = {}".format(self.value, self.depth))
        else:
            print("|   "*gap + "| ?(X({}) = {}):".format(self.feature, self.name))
            for child in self.child:
                child._print_(gap + 1)

class Node:
    
    def __init__(self, val=None, depth=None, column_req=None):
        self.value = val
        self.depth = depth
        self.column_req = column_req
        self.child = {}
        self.split_value=None
        if column_req is None:
            self.leaf = True
        else:
            self.leaf = False

    def node_val(self, X, max_depth = float('inf')):
        if self.leaf==True or self.depth>=max_depth:
            return self.value
        else:
                
            if X[self.column_req] < self.split_value:
                return self.child['left'].node_val(X, max_depth)
            else:
                return self.child['right'].node_val(X, max_depth)
            
    def _print_(self, indent=0):
        """
        Function to recursively print the entire tree structure
        """
        if self.leaf:
            print(' ' * indent + f'Predict: {self.value}')
        else:
            print(' ' * indent + f'[X{self.column_req} < {self.split_value}]')
            print(' ' * indent + f'Y:')
            self.child['left']._print_(indent + 2)
            print(' ' * indent + f'N:')
            self.child['right']._print_(indent + 2)



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root=None

    def build_tree(self, X: pd.DataFrame, y: pd.Series, cur_depth) -> None:
        """
        Function to construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        if cur_depth<self.max_depth and X.size > 1 and (max(list(X.nunique()))) > 1:
            # print(cur_depth)
            

            
                
            max_info_gain = -float("inf")
            split_val = None
            column_req = None
            for column in X:
                # print(column)
                attr = pd.Series(X[column])
                split, info_gain = information_gain(y, attr, self.criterion)
                if(info_gain > max_info_gain):
                    max_info_gain = info_gain
                    split_val = split
                    column_req = column

        
            # print(column_req, split_val)
            newnode = Node(column_req=column_req)
        

            # print(column_req)
            dict1 = {1:X[column_req], 2:y}
            dict1 = pd.concat(dict1, axis=1)
        

            # if split_val != False:
            newnode.split_value=split_val
            
            X_left = X[X[column_req]<split_val].reset_index(drop=True)
            X_right = X[X[column_req]>=split_val].reset_index(drop=True)


            y_left = (dict1[dict1[1] <= split_val]).iloc[:,-1].reset_index(drop=True)
            y_right = (dict1[dict1[1] > split_val]).iloc[:,-1].reset_index(drop=True)
            del X_left[column_req]
            del X_right[column_req]
            newnode.child.update({'left' : self.build_tree(X_left, y_left,cur_depth+1)})
            newnode.child.update({'right' : self.build_tree(X_right, y_right,cur_depth+1)})
            # print(self.build_tree(X_left, y_left, newnode, cur_depth+1))
            # print(X_left, y_left, newnode.value, cur_depth)
        
            if not check_ifreal(y):
                newnode.val = y.mode()[0]
            else:
                newnode.val = y.mean()
                
            newnode.depth = cur_depth
            return newnode

        else:
            if y.dtype.name == "category": 
                return Node(val=y.mode()[0], depth=cur_depth)
            else:
                return Node(val=y.mean(), depth=cur_depth)
            



    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Function to train the model
        """
        self.y = y
        self.X = X
        if (not check_ifreal(y) and any(check_ifreal(X[j]) for j in X)) or (check_ifreal(y) and any(check_ifreal(X[j]) for j in X)): # for Real input and Discrete/Real output
            self.root = self.build_tree(X, y, cur_depth = 0)
        elif (check_ifreal(y) and not any(check_ifreal(X[j]) for j in X)) or (not check_ifreal(y) and not any(check_ifreal(X[j]) for j in X)): # for Discrete input and real/Discrete output
            self.tree = Node1(X,y,self.max_depth,"root",crit = self.criterion)


    def predict(self, X: pd.DataFrame, max_depth=5) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        if (not check_ifreal(self.y) and any(check_ifreal(X[j]) for j in X)) or (check_ifreal(self.y) and any(check_ifreal(X[j]) for j in X)):
            y = []
            for ind in X.index:
                y.append(self.root.node_val(X.loc[ind], max_depth))
            return pd.Series(y)
        elif (check_ifreal(self.y) and not any(check_ifreal(X[j]) for j in X)) or (not check_ifreal(self.y) and not any(check_ifreal(X[j]) for j in X)):
            y_hat = []
            for _, row in X.iterrows():
                node1 = self.tree
                while node1.value is None:
                    feature_value = row[node1.feature]
                    for child in node1.child:
                        if child.name == feature_value:
                            node1 = child
                            break
                y_hat.append(node1.value)
            return pd.Series(y_hat)

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
        if not check_ifreal(self.y) and check_ifreal(self.X[0]):
            pass
        elif (check_ifreal(self.y) and not any(check_ifreal(self.X[j]) for j in self.X)) or (not check_ifreal(self.y) and not any(check_ifreal(self.X[j]) for j in self.X)):
            self.tree._print_()
        else:
            self.root._print_()
