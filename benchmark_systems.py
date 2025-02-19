
import numpy as np

"""
Provides a set of benchmark systems for testing the performance of the classical and pruned reservoir computing models
"""

def load_data(name: str) -> tuple :

    # 

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    return (X_train, y_train, X_test, y_test)


def split_data(X: np.ndarray, y: np.ndarray, train_size: float, shuffle: bool = True) -> tuple :

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    return (X_train, y_train, X_test, y_test)


def generate_lorentz_data(n_samples: int) -> tuple :

    return (X, y)


def generate_narma_data(n_samples: int) -> tuple:


    return (X, y)





