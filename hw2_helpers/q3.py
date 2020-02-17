from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
from math import exp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
# add constant one feature - no bias needed
x = np.concatenate((np.ones((506, 1)), x), axis=1)
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

# helper function


def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    # print(A.shape)
    # print(B.shape)
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    # print("dist: ", dist)
    return dist

# to implement


def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    # TODO
    dist = l2(np.array(x_train), test_datum.T)
    x = -dist/(2*(tau**2))
    diag_a = x - logsumexp(x)
    a = np.exp(diag_a)
    A = np.diag(a.flatten())
    # print("dist: ", np.array(dist).shape)
    # diag_a = []
    # for i in range(len(x_train)):
    #     log_a = dist[i] - logsumexp(dist)
    #     diag_a.append(exp(log_a))
    # A = np.diag(np.array(diag_a))

    x_train = np.array(x_train)
    xta = np.dot(x_train.T, A)
    LSH = np.dot(xta, x_train)+lam*np.identity(d)
    RHS = np.dot(xta, y_train)
    w = np.linalg.solve(LSH, RHS)

    y_hat = np.dot(w.T, test_datum)

    return y_hat


# helper function


def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i, :].reshape(d, 1), x_train, y_train, tau)
                                for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

# to implement


def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    # TODO
    x_parted = []
    y_parted = []
    part_len = len(x) / k
    part_len = int(part_len)
    cur = 0
    for i in range(k):
        x_parted.append(x[cur: cur+part_len])
        y_parted.append(y[cur: cur+part_len])
        cur += part_len

    losses = []
    for i in range(k):
        x_test = x_parted[i]
        y_test = y_parted[i]
        x_train = []
        y_train = []
        for j in range(k):
            if j != i:
                x_train.extend(x_parted[j])
                y_train.extend(y_parted[j])
        # print("len(x_test)", len(x_test))
        # print("len(y_test)", len(y_test))
        # print("len(x_train)", len(x_train))
        # print("len(y_train)", len(y_train))

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        losses.append(run_on_fold(x_test, y_test, x_train, y_train, taus))

    return np.array(losses)


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(losses)
    print("min loss = {}".format(losses.min()))
