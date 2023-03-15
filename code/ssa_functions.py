import numpy as np
import scipy
from sklearn import metrics

def create_hankel_matrix(s: np.array, emb_size: int, step_size: int = 1) -> np.array:
    '''
        Creates hankel matrix from time series

        s:  1d time sequence
        step_size: distance between 2 following intervals
        emb_size: 

        return: H [emb_size x (n - k * (e-1))/k]
    '''

    n = len(s)

    L = (n - step_size * (emb_size-1)) // step_size
    assert L > 0, "Too big step size"
    assert emb_size <= L, "Emb > width!"

    H = np.zeros(shape=(emb_size, L))
    for i in range(emb_size):
        for j in range(L):
            H[i, j] = s[(i + j) * step_size]
    
    return H


def get_eigenvalues_and_vectors(H: np.matrix) -> (np.array, np.matrix):
    '''
        returns eigenvalues and eigenvectors of S=H^TH

        H: [emb_size, len]
    '''

    S = H @ H.T

    _, eign_values, eign_vectors = scipy.linalg.svd(S)

    eign_vectors = eign_vectors.T # столбцы - собственные векторы

    # print(eign_vectors[:, 0][:, None].shape)

    X_s = np.array([eign_vectors[:, i][:, None] @ eign_vectors[:, i][:, None].T @ H for i in range(S.shape[0])]) # длина эмбеддинга
    return X_s, eign_vectors, eign_values


def hankelize(X: np.array) -> np.array:
    '''
        averaging of matrix X by anti-diagonals
    '''

    Y = np.zeros_like(X)

    for diag_num in range(X.shape[0] + X.shape[1]):
        diag_sum = 0
        diag_len = 0

        for i in range(0, X.shape[0]):
            j = diag_num - i
            if j < 0 or j >= X.shape[1]:
                continue
            
            diag_sum += X[i, j]
            diag_len += 1
        
        for i in range(0, X.shape[0]):
            j = diag_num - i
            if j < 0 or j >= X.shape[1]:
                continue

            Y[i, j] = diag_sum / diag_len
    return Y


def group_by_eign_incr_and_hank(X_s: np.array, eign_values: np.array, k: int) -> np.array:
    '''
        Sum all elements first and then hankelize!
        So, we use 1 large group [1, ..., k]

        X_s = [X_1, ..., X_emb_size] - components of svd decomposition. 1-rank matrices
        eign_values - corresponding to X_s
        k - number of first components used for summation
    '''
    return hankelize(X_s[:k].sum(axis=0))


def flatten_time_serie(H: np.array) -> np.array:
    '''
        unpack hankel matrix of time serie H to a sime serie
    '''

    s = list(H[0, :]) + list(H[1:, -1])
    return np.array(s)


def mse_loss(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

def mae_loss(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

def mape_loss(y_true, y_pred):
    return metrics.mean_absolute_percentage_error(y_true, y_pred)