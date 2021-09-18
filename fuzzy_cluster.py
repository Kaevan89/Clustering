import numpy as np
from cluster import Cluster


class FuzzyCluster(Cluster):
    """
    Class that implements the fuzzy clustering algorithm
    """

    def __init__(self, data: np.ndarray, centers_number: int, m: float = 2., seed: int = None):
        """
        Main constructor

        Warning! A number of centers close to the number of data does not produce good results in this implementation

        :param data: ndarray, the data set to be clustered
        :param centers_number: int, number of classes wished
        :param m: float, fuzzy coefficient
        :param seed: int, number to set the random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        super().__init__(data, np.random.rand(centers_number, data.shape[1]))
        self.m = m
        self.membership_values = None

    def _calc_membership(self, point, center):
        return 1 / (np.sum([self.dist(point, center) /
                            self.dist(point, center_iterator)
                            for center_iterator in self.centers]) ** (1 / (self.m - 1)))

    def _calc_new_centers(self):
        return np.array(
            [np.sum([self._calc_membership(point, center) ** self.m * point for point in self.data], axis=0) /
             np.sum([self._calc_membership(point, center) ** self.m for point in self.data]) for center in
             self.centers])

    def calc_centers(self, iterations: int = 100, error: float = 1E-6):
        """
        Fit method, update the centers using fuzzy clustering

        :param iterations: int, number to set the max number of iterations
        :param error: float, set the allowed tolerance
        """
        for _ in range(iterations):
            new_centers = self._calc_new_centers()
            self.centers = new_centers
            if np.allclose((self.centers, new_centers), error):
                break

    def calc_membership_values(self, return_values: bool = True) -> np.ndarray:
        """
        Method to calc the membership values

        :param return_values: bool, by default True, set it in false if you don't want keep values.
        :return: ndarray with membership values
        """
        self.membership_values = np.array([[self._calc_membership(x, v) for v in self.centers] for x in self.data])
        if return_values:
            return self.membership_values

    def classify(self) -> np.ndarray:
        """
        Method to get the classification

        :return: ndarry that contains the class of every point
        """
        self.labels = np.argmax(self.calc_membership_values(), axis=1)
        return self.labels

    def calc_PC(self):
        """
        Method to calc the partition coefficient defined by Bezdek
        """
        if self.membership_values is None:
            self.calc_membership_values(False)
        return np.sum(np.square(self.membership_values)) / self.data.shape[0]

    def calc_CE(self):
        """
        Method to calculate the lack of fuzivity of the groups
        """
        if self.membership_values is None:
            self.calc_membership_values(False)
        return -np.sum(self.membership_values * np.log10(self.membership_values)) / self.data.shape[0]

    def calc_I(self):
        """
        Method to calc the safety of a classification
        """
        if self.membership_values is None:
            self.calc_membership_values(False)
        max_values = np.max(self.membership_values, axis=1, keepdims=True)
        lambda_i = max_values - self.membership_values
        C = self.m - 1
        return np.sum(lambda_i * np.exp(lambda_i), axis=1) / (C * max_values * np.exp(max_values)).reshape(-1)
