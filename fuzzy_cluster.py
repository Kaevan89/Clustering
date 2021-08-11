import numpy as np
from cluster import Cluster


class FuzzyCluster(Cluster):
    '''
    Class that implements the fuzzy clustering algorithm
    '''

    def __init__(self, data: np.ndarray, centers_number: int, m: float = 2., seed: int = None):
        '''
        Main constructor

        Warning! A number of centers close to the number of data does not produce good results in this implementation

        :param data: ndarray, the data set to be clustered
        :param centers_number: int, number of classes wished
        :param m: fuzzy coefficient
        :param seed: int number, set the random seed for reproducibility
        '''
        if seed is not None:
            np.random.seed(seed)
        super().__init__(data, np.random.rand(centers_number, data.shape[1]))
        self.m = m

    def _calc_membership(self, point, center):
        return 1 / (np.sum([self.dist(point, center) /
                            self.dist(point, center_iterator)
                            for center_iterator in self.centers]) ** (1 / (self.m - 1)))

    def _calc_new_centers(self):
        return np.array(
            [np.sum([self._calc_membership(point, center) ** self.m * point for point in self.data], axis=0) /
             np.sum([self._calc_membership(point, center) ** self.m for point in self.data]) for center in self.centers])

    def calc_centers(self, iterations: int=100, error: float=1E-6):
        '''
        Fit method, update the centers using fuzzy clustering

        :param iterations: int, number to set the max number of iterations
        :param error: float, set the allowed tolerance
        '''
        for _ in range(iterations):
            new_centers = self._calc_new_centers()
            if np.allclose((self.centers, new_centers), error):
                self.centers = new_centers
                break
            self.centers = new_centers

    def calc_membership_values(self) -> np.ndarray:
        '''
        Method to calc the membership values

        :return: ndarray with membership values
        '''
        return np.array([[self._calc_membership(x, v) for v in self.centers] for x in self.data])

    def classify(self) -> np.ndarray:
        '''
        Method to get the classification

        :return: ndarry that contains the class of every point
        '''
        self.labels = np.argmax(self.calc_membership_values(), axis=1)
        return self.labels
