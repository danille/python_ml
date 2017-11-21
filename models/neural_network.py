import numpy as np

class Perceptron(object):
    """
    Perceptron classifier

    Parameters
    ------------
    eta: Learning rate (between 0.0 and 1.0)
    n_epochs: number of passes over the training dataset
    random_state: random number generator seed for random weight initialization

    Attributes
    ------------
    w_: 1d-array weights after fitting
    errors_: number of misclassifications (updates) in each epoch
    """

    def __init__(self, eta: float=0.01, n_epochs: int=50, random_state: int=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y) -> object:
        """

        :param X: {array-like} shape = [n_samples, m_features]
                Training vectors, where n_samples is number of samples and m_features is number of features
        :param y: array-like, shape = [n_samples] Target values
        :return: self
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        :param X: Training vectors
        :return: activation for each training example
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Make prediction, return class label after unit step
        :param X: Training vectors
        :return: predictions shape = [n_samples, 1]
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)