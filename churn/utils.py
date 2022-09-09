from types import new_class
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # Credit Fraud dataset has 2 classes
    n_features = 20  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

""" Read and split data """

def load_data() -> Dataset:    
    data = pd.read_csv('data1.csv')
    data.reset_index(drop=True)
    df =np.array(data)
    X = df[:,:-1]
    y =df[:,-1]
    
    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, y_train = X[:1262], y[:1262]
    x_test, y_test = X[1262:], y[1262:]
    return (x_train, y_train), (x_test, y_test)
        

""" Read data for the other client """
def load_data_client() -> Dataset:
    data = pd.read_csv('data2.csv')
    data.reset_index(drop=True)
    df =np.array(data)
    X = df[:,:-1]
    y =df[:,-1]
    
    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, y_train = X[:4182], y[:4182]
    x_test, y_test = X[4182:], y[4182:]
    return (x_train, y_train), (x_test, y_test)
def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )