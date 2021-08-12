import numpy as np


class Cluster():
    '''
    Abstract implement of cluster class
    '''

    def __init__(self, data: np.ndarray, centers: np.ndarray):
        '''
        Main constructor of cluster class with random init center
        
        :param data: ndarray, the data set to be clustered
        :param centers: ndarray, centers of Clusters
        '''
        self.data = data
        self.centers = centers
        self.labels = None

    @staticmethod
    def dist(point_1: np.ndarray, point_2: np.ndarray) -> np.ndarray:
        '''
        Method to calc the square euclidean distanc
        
        :param point_1: ndarray with shape (n, m) that contains first points
        :param point_2: ndarray with shape (n, m) that contains second points
        :return: ndarray with shape (n, ) of euclidean distances
        '''
        return np.sum((point_1 - point_2) ** 2)

    def set_centers(self, centers: np.ndarray):
        '''
        Change the centers values
        
        :param centers: ndarray new centers values gives by develop
        '''
        if self.centers.shape[1] != centers.shape[1]:
            raise "No valid centers"
        self.centers = centers

    def calc_SSW(self) -> float:
        '''
        Method to calc the SSW coefficient

        :return: float, SSW coefficient
        '''
        if self.labels is None:
            raise "No labels allocated, first call classify method"
        SSW = 0
        for index, center in enumerate(self.labels):
            SSW += Cluster.dist(self.centers[center], self.data[index])
        return SSW

    def calc_TSS(self) -> float:
        '''
        Method to calc the TSS coefficient

        :return: float, TSS coefficient
        '''
        return float(np.sum(self.dist(self.data, self.data.mean(0))))

    def calc_SSB(self) -> float:
        '''
        Method to calc the SSB coefficient

        :return: float, SSB coefficient
        '''
        return self.calc_TSS() - self.calc_SSW()

    def get_metrics(self) -> dict:
        '''
        Method to get SSW, TSS and SSB

        :return: JSON with SSW, TSS, SSB
        '''
        SSW = self.calc_SSW()
        TSS = self.calc_TSS()
        return {"SSW": SSW, "TSS": TSS, "SSB": TSS - SSW}
