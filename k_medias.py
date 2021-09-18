import random
import numpy as np
from cluster import Cluster


class KCluster(Cluster):

    def __init__(self, data: np.ndarray, centers_number: int, seed: int = None):
        """
        Main constructor of KCluster class with random init centers

        :param data: ndarray, the data set to be clustered
        :param centers_number: int, number of classes wished
        :param seed: int number, set the random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        super().__init__(data, data[tuple([random.sample(range(data.shape[0]), centers_number)])])

    def _calc_dist_matrix(self):
        return np.array([[self.dist(point, center) for point in self.data] for center in self.centers])

    def _calc_group_matrix(self, all_distances):
        point_classes = np.argmin(all_distances, axis=0)
        group_matrix = np.zeros((self.centers.shape[0], self.data.shape[0]), dtype=int)
        for point, point_class in enumerate(point_classes):
            group_matrix[point_class, point] = 1
        return np.array(group_matrix)

    def calc_centers(self, iterations: int = 100):
        """
        Fit method, update the centers using K Medias method

        :param iterations: int, number to set the max number of iterations
        """
        old_group_matrix = np.zeros((self.centers.shape[0], self.data.shape[0]), dtype=int)
        for _ in range(iterations):
            all_distances = self._calc_dist_matrix()
            group_matrix = self._calc_group_matrix(all_distances)
            self.centers = (group_matrix @ self.data) / np.sum(group_matrix, axis=1, keepdims=True)
            if np.array_equal(group_matrix, old_group_matrix):
                break
            old_group_matrix = group_matrix

    def classify(self) -> np.ndarray:
        """
        Method to get the classification

        :return: ndarry that contains the class of every point
        """
        all_distances = self._calc_dist_matrix()
        group_matrix = self._calc_group_matrix(all_distances).T
        self.labels = np.array([pertinence.nonzero()[0].item() for pertinence in group_matrix])
        return self.labels
