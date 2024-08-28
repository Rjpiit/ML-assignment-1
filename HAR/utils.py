import pandas as pd
import math
import numpy as np


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # Entropy is the sum of the negative probability times the log of the probability for each class
    # Values are the distinct classes, counts are the number of instances of each class
    value, counts = np.unique(Y, return_counts=True)
    # Probabilities are the fraction of instances of each class
    probabilities = counts / len(Y)
    entropy = 0  # initialising the value of entropy to zero
    for probability in probabilities:
        entropy -= float(probability) * math.log(probability, 2)
    # We have defined entropy in log base 2 as taught in class
    return entropy
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # Gini index is 1 - the sum of the probability squared for each class
    # Values are the distinct classes, counts are the number of instances of each class
    value, counts = np.unique(Y, return_counts=True)
    probabilities = counts / len(Y)
    gini_index = 1
    gini_index -= sum(probability ** 2 for probability in probabilities)
    return gini_index
    pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: "information_gain"):
    """
    Function to calculate the information gain
    """
    # creating a dataframe with the column having the first output and the second column having the values of the attribute series
    dict1 = {1: Y, 2: attr}
    df = pd.concat(dict1, axis=1)

    # in the case of Discrete input and discrete output, and information gain as the criterion
    if(Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "information_gain"):
        uni_attr = attr.unique()  # collection of unique attribute values
        entro1 = entropy(Y)  # entropy of the total dataset
        entro2 = 0
        for attri in uni_attr:
            # spilting the dataframe according to the attribute values
            df1 = df[df[2] == attri]
            # creating a series with the respective outputs of the selected attribute
            y_ = df1.iloc[:, 0].reset_index(drop=True)
            entro2 += entropy(y_)*(len(y_))  # adding to the entropy
        return((False, entro1 - (entro2/len(Y))))
    # in the case of Discrete input and discrete output, and gini index as the criterion
    elif(Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "gini_index"):
        uni_attr = attr.unique()
        gini_1 = gini_index(Y)  # Gini Index of the total dataset
        gini_2 = 0  # initialising the value of the gini index of the sub tables to zero
        for attri in uni_attr:
            df1 = df[df[2] == attri]
            # creating a series with the output of the corresponding attribute value
            y_ = df1.iloc[:, 0].reset_index(drop=True)
            gini_2 += gini_index(y_)*(len(y_))  # updating the gini_index
        return((False, gini_1 - (gini_2/len(Y))))
    # in case of a real input and discrete output, and gini_index as the measure
    elif(Y.dtype.name == "category" and attr.dtype.name != "category" and criterion != "information_gain"):
        # sorting the dataframe according to the attribute values
        sor_dafram = df.sort_values(2).reset_index(drop=True)
        best_split = 0
        max_val = -np.inf  # initialising the max_value
        # iterating through the attribute values in order to find the split which gives the maximum gain
        for ind in sor_dafram.index:
            if ind == 0:
                continue
            split = float(sor_dafram[2][ind] + sor_dafram[2][ind-1])/2
            # spliting the dataframe into 2 based on the split value and making the corresponding output series
            df1 = sor_dafram[sor_dafram[2] <= split].reset_index(drop=True)
            df2 = sor_dafram[sor_dafram[2] > split].reset_index(drop=True)
            y_1 = df1.iloc[:, 0].reset_index(drop=True)
            y_2 = df2.iloc[:, 0].reset_index(drop=True)
            gini_1 = (gini_index(y_1)*(len(y_1)) +
                      gini_index(y_2)*(len(y_2)))/len(Y)
            val = gini_index(Y) - gini_1
            if(val > max_val):
                max_val = val
                best_split = split
        return((best_split, max_val))
    # in case of real input and discrete output, and entropy measure
    elif(Y.dtype.name == "category" and attr.dtype.name != "category" and criterion == "information_gain"):
        sor_dafram = df.sort_values(2).reset_index(drop=True)
        best_split = 0
        max_val = -np.inf  # initialising the max_value
        # iterating through the attribute values in order to find the split which gives the maximum gain
        for ind in sor_dafram.index:
            if ind == 0:
                continue
            # spliting the dataframe into 2 based on the split value and making the corresponding output series
            split = float(sor_dafram[2][ind] + sor_dafram[2][ind-1])/2
            df1 = sor_dafram[sor_dafram[2] <= split].reset_index(drop=True)
            df2 = sor_dafram[sor_dafram[2] > split].reset_index(drop=True)
            y_1 = df1.iloc[:, 0].reset_index(drop=True)
            y_2 = df2.iloc[:, 0].reset_index(drop=True)
            entro_1 = (entropy(y_1)*(len(y_1)) +
                       entropy(y_2)*(len(y_2)))/len(Y)
            val = entropy(Y) - entro_1
            if(val > max_val):
                max_val = val
                best_split = split
        return((best_split, max_val))
    # in case of Discrete input and real output, variance will be used in this case
    elif(Y.dtype.name != "category" and attr.dtype.name == "category"):
        uni_attr = attr.unique()
        var_1 = np.var(Y)  # Gini Index of the total dataset
        var_2 = 0
        # iterating and finding the variance of each subtable of the corresponding attribute
        for attri in uni_attr:
            df1 = df[df[2] == attri]
            y_ = df1.iloc[:, 0].reset_index(drop=True)
            var_2 += (np.var(y_))*(len(y_))
        return((False, var_1 - (var_2/len(Y))))
    # in case of real input and real output, variance will be used as a measure
    elif(Y.dtype.name != "category" and attr.dtype.name != "category"):
        sor_dafram = df.sort_values(2).reset_index(drop=True)
        best_split = 0
        max_val = -np.inf
        # iterating through the index values to find the best split
        for ind in sor_dafram.index:
            if ind == 0:
                continue
            split = float((sor_dafram[2][ind] + sor_dafram[2][ind-1])/2)
            y_1 = sor_dafram[sor_dafram[2] <=
                             split].iloc[:, 0].reset_index(drop=True)
            y_2 = sor_dafram[sor_dafram[2] >
                             split].iloc[:, 0].reset_index(drop=True)
            var_2 = (np.var(y_1)*(len(y_1)) + np.var(y_2)*(len(y_2)))/len(Y)
            val = np.var(Y) - var_2
            if(val > max_val):
                max_val = val
                best_split = split
        return((best_split, max_val))
    pass
