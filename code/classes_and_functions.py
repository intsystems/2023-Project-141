import numpy as np
from numpy import linalg
import pandas as pd
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib

import seaborn as sns
sns.set(font_scale=1.3, palette='Set2')

# обратите внимание, что Scikit-Learn импортируется как sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import scipy.stats as sps
import scipy

from tqdm import tqdm_notebook
import torch
import torch.nn as nn
import geotorch

# Datasets
from torch.utils.data import Dataset
from torchvision import datasets

# On top of dataset for batching, shuffle, ...
from torch.utils.data import DataLoader

def generate_noised_sin(sample_size, window_scale=1, noise_std=0.2, visualize=False):
    x_min, x_max = -np.pi * window_scale, np.pi * window_scale
    
    X = np.linspace(x_min, x_max, sample_size)
    y_true = np.sin(X)
    y = y_true + sps.norm(loc=0, scale=noise_std).rvs(sample_size)

    if visualize:
        plt.figure(figsize=(7, 5))
        plt.scatter(X, y, c='b', label='noised sin')
        plt.plot(X, y_true, color='r', linewidth=3, label='true sin')
        plt.title('$y = sin(x) + \\varepsilon$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend();

    return X, y_true, y

class TimeSerie_1d_Dataset(Dataset):
    def __init__(self, in_len: int, out_len: int, s_noice: np.array, s_true: np.array = None):
        '''
            s = [s_1, s_2, ... ]- 1d time serie
            in_len - length of sequence of following time measurements
            out_len - length of predictions (the last of in_len elements)

            ds[1] = [s_1, ..., s_k]
        '''

        assert s_noice.shape == s_true.shape

        self.s_noice = torch.FloatTensor(s_noice)
        self.s_true = torch.FloatTensor(s_true)
        self.in_len = in_len
        self.out_len = out_len
    
    def __len__(self):
        return len(self.s_noice) - (self.in_len - 1) - max(0, self.out_len - self.in_len)
    
    def __getitem__(self, idx):
        '''
            the 0-th element has index max(out_len - in_len, 0)
        '''
        assert idx >= 0, 'idx [left] out of range'
        assert idx < len(self), 'idx [right] out of range'
        
        right_idx = idx + max(0, self.out_len - self.in_len)

        idx = right_idx

        return self.s_noice[idx:idx + self.in_len], \
               self.s_noice[idx + self.in_len - self.out_len: idx + self.in_len], \
               self.s_true[idx + self.in_len - self.out_len: idx + self.in_len]

        # assert idx + self.in_len <= len(self.s), 'idx [right] out of range'
        # assert idx + self.in_len - self.out_len >= 0, 'idx [left] out of range'
        # assert idx >= 0, 'idx [zero] out of range'
        # return self.s[idx:idx + self.in_len], self.s[idx + self.in_len - self.out_len, idx + self.in_len]