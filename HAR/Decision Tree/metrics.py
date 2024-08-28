from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    # The number of predictions should be equal to the number of ground truth labels
    assert y_hat.size == y.size
    # TODO: Write here
    n = y.size
    true_positive = 0
    for i in range(n):
        if y_hat[i] == y[i]:
            true_positive += 1
    # Corner case of dataset size = 0
    if(n == 0):
        return 1
    # Accuracy is the fraction of correct predictions
    return float(true_positive) / n
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    # The number of predictions should be equal to the number of ground truth labels
    assert y_hat.size == y.size
    n = y.size
    true_positive = 0
    y_hat_positive = 0
    for i in range(n):
        if y_hat[i] == cls:
            y_hat_positive += 1
            if y_hat[i] == y[i]:
                true_positive += 1
    if(n == 0 or y_hat_positive == 0):
        return 1
    # Precision is the fraction of relevant instances among the retrieved instances
    return float(true_positive) / y_hat_positive
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    # The number of predictions should be equal to the number of ground truth labels
    assert y_hat.size == y.size
    n = y.size
    true_positive = 0
    y_positive = 0
    for i in range(n):
        if y[i] == cls:
            y_positive += 1
            if y_hat[i] == y[i]:
                true_positive += 1
    if(n == 0 or y_positive == 0):
        return 1
    # Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances
    return float(true_positive) / y_positive
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    # The number of predictions should be equal to the number of ground truth labels
    assert y_hat.size == y.size
    n = y.size
    sq_error = 0
    for i in range(n):
        sq_error += (y_hat[i] - y[i]) ** 2
    mean_sq_error = float(sq_error) / n
    # Root mean squared error is the square root of the mean of the squared errors
    return mean_sq_error ** 0.5
    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    # The number of predictions should be equal to the number of ground truth labels
    assert y_hat.size == y.size
    abs_error = 0
    n = y.size
    for i in range(n):
        abs_error += abs(y_hat[i] - y[i])

    # Mean absolute error is the mean of the absolute errors
    return float(abs_error) / n
    pass
