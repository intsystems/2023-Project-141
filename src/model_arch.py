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

import torch
import torch.nn as nn
import geotorch

# Datasets
from torch.utils.data import Dataset
from torchvision import datasets

# On top of dataset for batching, shuffle, ...
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import trange

class NN_2_orthogonal(nn.Module):
    def __init__(self, in_len, hid_dim, out_len):
        '''
        '''
        super().__init__()

        # One line suffices: Instantiate a linear layer with orthonormal columns
        self.W1 = nn.Linear(in_len, hid_dim, bias=True)
        geotorch.orthogonal(self.W1, "weight")
        self.act1 = nn.ReLU()

        self.W2 = nn.Linear(hid_dim, out_len, bias=True)
        # geotorch.orthogonal(self.W2, "weight")
        self.act2 = nn.Identity()

    def forward(self, x):
        # self.linear is orthogonal and every 3x3 kernel in self.cnn is of rank 1
        x = self.W1(x)
        x = self.act1(x)

        x = self.W2(x)
        x = self.act2(x)

        return x

def model_num_params(model):
    sum_params = 0
    for param in model.named_parameters():
        num_params = np.prod(param[1].shape)
        print('{: <19} ~  {: <7} params'.format(param[0], num_params))
        sum_params += num_params
    print(f'\nIn total: {sum_params} params')
    return sum_params


def train_one_epoch(model, optimizer, loader, criterion, device='cpu'):
    model.train()
    losses_tr = []
    losses_train_true = []

    for batch, y, y_true in loader:
        batch = batch.to(device)
        y = y.to(device)
        y_true = y_true.to(device)
        # batch = [batch_size, seq_len]

        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        losses_tr.append(loss.item()) 

        loss_true = criterion(out, y_true)
        losses_train_true.append(loss_true.item())

    return model, optimizer, np.mean(losses_tr), np.mean(losses_train_true)


def validate(model, loader, criterion, device='cpu'):
    model.eval()

    losses = []
    losses_true = []

    with torch.no_grad():
        for batch, y, y_true in loader:
            batch = batch.to(device)
            y = y.to(device)
            y_true = y_true.to(device)

            out = model(batch)
            loss = criterion(out, y)

            losses.append(loss.item())

            loss_true = criterion(out, y_true)
            losses_true.append(loss_true.item())
    
    return np.mean(losses), np.mean(losses_true)


def learning_loop(model, optimizer, loader, criterion, epochs=10, val_every=1, draw_every=1, separate_show=False, device='cpu'):
    losses = {'train': [], 'val': []}
    losses_true = {'train': [], 'val': []}

    for epoch in range(1, epochs+1):
        # print(f'#{epoch}/{epochs}:')
        model, optimizer, loss, loss_true = train_one_epoch(model, optimizer, loader, criterion, device)
        losses['train'].append(loss)
        losses_true['train'].append(loss_true)

        if not (epoch % val_every):
            loss, loss_true = validate(model, loader, criterion, device)
            losses['val'].append(loss)
            losses_true['val'].append(loss_true)

        if not (epoch % draw_every):
            clear_output(True)
            fig, ax = plt.subplots(1, 2 if separate_show else 1, figsize=(12, 6))
            fig.suptitle(f'#{epoch}/{epochs}:')

            if separate_show:
                plt.subplot(121)
                plt.title('loss on train')
            plt.plot(losses['train'], 'r.-', linewidth=2, label='train-noice')
            plt.plot(losses_true['train'], 'g.-', linewidth=2, label='train-true')

            plt.xlabel('epoch')
            plt.ylabel('mse')
            plt.legend()

            if separate_show:
                plt.subplot(122)
                plt.title('loss on validation')
            else:
                plt.title('losses')
            plt.plot(losses['val'], 'r.-', linewidth=2, label='val-noice')
            plt.plot(losses_true['val'], 'g.-', linewidth=2, label='val-true')

            plt.xlabel('epoch')
            plt.ylabel('mse')
            plt.legend()
            
            plt.show()
    
    return model, optimizer, losses, losses_true