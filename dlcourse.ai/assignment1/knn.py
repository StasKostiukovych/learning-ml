import numpy as np
import scipy
from collections import Counter

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)


        return self.predict_labels(dists)

    def compute_distances_two_loops(self, X):

        num_train = self.train_X.shape[0]
        num_test = X.shape[0]


        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = np.sum(abs(X[i_test] - self.train_X[i_train]))
        return dists


    def compute_distances_one_loop(self, X):

        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = (np.sum(abs(np.matlib.repmat(X[i_test], num_train, 1) - self.train_X), axis=1))

        return dists

    def compute_distances_no_loops(self, X):

        num_train = self.train_X.shape[0]
        num_test = X.shape[0]

        dists = scipy.spatial.distance.cdist(X, self.train_X, 'minkowski', p=1)

        return dists



    def predict_labels(self, dists):

        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):

            top_k_indx = np.argsort(dists[i])[:self.k]
            closest_y = self.train_y[top_k_indx]

            vote = Counter(closest_y)
            count = vote.most_common()
            pred[i] = count[0][0]

        return pred
