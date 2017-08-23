# Stochastic Subgradient (SSG) Method for Averaging Time Series
# under Dynamic Time Warping (DTW).
#
# Translation by Khaled Sakallah, based on the Matlab code
# of the SSG algorithm in https://doi.org/10.5281/zenodo.216233
# Original Author: David Schultz, DAI-Lab, TU Berlin, Germany, 2017
####################################################################

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from tqdm import tqdm


def ssg(X, n_epochs=None, eta=None, init_sequence=None, return_f=False):
    # Inputs
    # X is a 3-dim matrix consisting of possibly multivariate time series.
    #   dim 1 runs over the sample time series
    #   dim 2 runs over the length of a time series
    #   dim 3 runs over the dimension of the datapoints of a time series
    #
    # Optional Inputs
    # n_epochs        is the number of epochs
    # eta             is a vector of step sizes, eta(i) is used in the i-th update
    # init_sequence   if None  --> use a random sample of X
    #                 if > 0   --> use X[init_sequence]
    #                 if <= 0  --> use medoid of X
    #                 if it is a time series --> use it
    # return_f        if True  --> Frechet variations for each epoch are returned
    #
    # Outputs
    # z               the solution found by SSG (an approximate sample mean under dynamic time warping)
    # f               Vector of Frechet variations. Is only returned if return_f=True
    
    N = X.shape[0]  # number of samples
    d = X.shape[2]  # dimension of data

    if n_epochs is None:
        n_updates = 1000
        n_epochs = int(np.ceil(n_updates / N))

    if eta is None:
        eta = np.linspace(0.1, 0.005, N)

    # initialize mean z
    if init_sequence is None:
        z = X[np.random.randint(N)]

    elif init_sequence > 0:
        z = X[int(init_sequence)]

    elif init_sequence <= 0:
        z = medoid_sequence(X)

    if return_f:
        f = np.zeros(n_epochs + 1)
        f[0] = frechet(z, X)

    # stochastic subgradient optimization
    with tqdm(total=n_epochs * N) as pbar:
        for k in range(1, n_epochs + 1):
            perm = np.random.permutation(N)
            for i in range(1, N + 1):
                pbar.update(1)
                x_i = X[perm[i - 1]]
                _, p = dtw(z, x_i, path=True)

                W, V = get_warp_val_mat(p)
                
                subgradient = 2 * (V * z - W.dot(x_i))

                c = (k - 1) * N + i
                if c <= eta.shape[0]:
                    lr = eta[c - 1]
                else:
                    lr = eta[-1]

                # update rule
                z = z - lr * subgradient

            if return_f:
                f[k] = frechet(z, X)

    if return_f:
        f = f[0:n_epochs + 1]
        return z, f

    else:
        return z


def dtw(x, y, path=False):
    # Local Variables: C, d, C_diag, k, C_d, m, N, p, C_r, y, x, n, D
    # Function calls: pdist2, min, cumsum, M, nargout, sqrt, zeros, dtw, size
    # %DTW dynamic time warping for multidimensional time series
    # %
    # % Input
    # % x:  [n x d]               d dimensional time series of length n
    # % y:  [m x d]               d dimensional time series of length m
    # %
    # % Output
    # % d:  [1 x 1]               dtw(x,y) with local Euclidean distance
    # % p:  [L x 2]   (optional)  warping path of length L
    # %
    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    N, d = x.shape
    M, _ = y.shape
    D = cdist(x, y) ** 2
    
    C = np.zeros((N, M))
    C[:, 0] = np.cumsum(D[:, 0])
    C[0, :] = np.cumsum(D[0, :])
    
    for n in range(1, N):
        for m in range(1, M):
            C[n, m] = D[n, m] + min(C[n - 1, m - 1], C[n - 1, m], C[n, m - 1])

d = np.sqrt(C[N - 1, M - 1])
    
    # % compute warping path p
    if path:
        n = N - 1
        m = M - 1
        p = np.zeros((N + M - 1, 2))
        p[-1, :] = (n, m)
        k = 1
        
        while n + m > 0:
            if n == 0:
                m = m - 1
            elif m == 0:
                n = n - 1
        
            else:
                C_diag = C[n - 1, m - 1]
                C_r = C[n, m - 1]
                C_d = C[n - 1, m]
                if C_diag <= C_r:
                    if C_diag <= C_d:
                        n = n - 1
                        m = m - 1
                    else:
                        n = n - 1
            
                elif C_r <= C_d:
                    m = m - 1

    else:
        n = n - 1
            
            p[-1 - k, :] = (n, m)
            k = k + 1

        p = p[-1 - k + 1:, :]
        
return d, p
    
    return d


def frechet(x, X):
    # Local Variables: dist, f, i, N, X, x
    # Function calls: Frechet, length, dtw
    N = X.shape[0]
    f = 0
    for i in range(N):
        dist = dtw(x, X[i])
        f = f + dist ** 2

    f = f / N
    return f


def medoid_sequence(X):
    # Local Variables: f, i, f_min, N, i_min, X, x
    # Function calls: Frechet, length, medoidSequence, inf
    # MEDOIDSEQUENCE returns medoid of X
    #  A medoid is an element of X that minimizes the Frechet function
    #  among all elements in X
    N = X.shape[0]
    f_min = np.inf
    i_min = 0
    for i in range(N):
        f = frechet(X[i], X)
        if f < f_min:
            f_min = f
            i_min = i

    x = X[i_min]
    return x


def get_warp_val_mat(p):
    # Local Variables: m, L, n, p, W, V
    # Function calls: length, ones, sparse, getWarpingAndValenceMatrix, sum
    #  W is the (sparse) warping matrix of p
    #  V is a vector representing the diagonal of the valence matrix
    L = p.shape[0]
    N = int(p[-1, 0]) + 1
    M = int(p[-1, 1]) + 1
    W = coo_matrix((np.ones(L), (p[:, 0], p[:, 1])), shape=(N, M)).toarray()
    V = np.sum(W, axis=1, keepdims=True)
    return W, V
