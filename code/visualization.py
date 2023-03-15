from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib

import scipy.stats as sps
import scipy


def visualize_predictions(X, y, y_true, model, dataset, batch_size, device='cpu'):
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions = []

        for i in range(len(dataset)):
            X_, y_, y_true_ = dataset[i]
            if i == 0:
                predictions += list(model(X_.to(device)).detach().cpu().numpy().reshape(-1))
            else:
                predictions += list(model(X_.to(device)).detach().cpu().numpy().reshape(-1))[-1:]

        #for i, (batch, _, _1) in enumerate(loader):
        #    batch = batch.to(device)
        #    predictions += list(model(batch).detach().cpu().numpy().reshape(-1))
        
        predictions = np.array(predictions)

    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, c='b', label='noised sin')
    plt.plot(X, y_true, color='r', linewidth=3, label='true sin')
    #for i in range(X.shape[1]):
    plt.plot(X, predictions, color='g', linewidth=3, label='recovered sin')
    #plt.title('$y = sin(x) + \\varepsilon$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend();


def plot_3d_gaussians(mu_s, Sigma_s, figsize=(10, 10), alpha=0.2, path_to_save=None, img_name=None):
    '''
        mu_s = [mu_1, ..., mu_n]; mu_i = (x_i, y_i, z_i) <- mean vectors
        Sigma_s <- variance matrices
        alpha - radius reduce factor
    '''

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # number of ellipsoids 
    ellipNumber = len(mu_s)

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=ellipNumber)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    #compute each and plot each ellipsoid iteratively
    for indx in range(ellipNumber):

        # your ellispsoid and center in matrix form
        A = Sigma_s[indx]
        center = mu_s[indx] 

        ellipsoid = (A @ np.stack((x, y, z), 0).reshape(3, -1) * alpha + center.reshape(3, -1)).reshape(3, *x.shape)

        ax.plot_surface(*ellipsoid,  rstride=2, cstride=2, \
                        color=m.to_rgba(indx), linewidth=0.1, alpha=0.4, shade=False)
    

    # ax.set_title(f'Neurons confidence areas')
    ax.set_xlabel(f'$x$', fontsize=20)
    ax.set_ylabel(f'$y$', fontsize=20)
    ax.set_zlabel(f'$z$', fontsize=20)

    ax.view_init(azim=45)

    if path_to_save is not None:
        plt.savefig(path_to_save + '/' + img_name + '.jpg')

    plt.show()


def visualize_gaussian_mixture_2d(mu_s, Sigma_s, figsize=(10, 10), N=50, path_to_save=None, img_name=None):
    '''
        plots mixture of 2d gaussian distribution on 3d plot
    '''

    distr_num = len(mu_s)
    rand_variables = [sps.multivariate_normal(mu_s[i], Sigma_s[i]) for i in range(distr_num)]

    # найду самую большую дисперсию по x и y. И найду крайние координаты центровю

    xvar_max = np.max(Sigma_s, axis=0)[0, 0]
    yvar_max = np.max(Sigma_s, axis=0)[1, 1]

    x_max = np.max(mu_s, axis=0)[0]
    x_min = np.min(mu_s, axis=0)[0]
    
    y_max = np.max(mu_s, axis=0)[1]
    y_min = np.min(mu_s, axis=0)[1]

    # creating grid
    dx = np.sqrt(xvar_max)
    dy = np.sqrt(yvar_max)

    dx = xvar_max
    dy = yvar_max

    x_s = np.linspace(x_min - 3 * dx, x_max + 3 * dx, N)
    y_s = np.linspace(y_min - 3 * dy, y_max + 3 * dy, N)
    X_s, Y_s = np.meshgrid(x_s, y_s)

    pos = np.dstack((X_s, Y_s))

    # pdf function
    PDF_s = np.array([
        rand_variables[i].pdf(pos) for i in range(len(rand_variables))
    ]).sum(axis=0) / distr_num # normalizing coefficient


    # colors
    norm = plt.Normalize(PDF_s.min(), PDF_s.max())
    colors = cm.jet(norm(PDF_s))

    # plot

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X_s, Y_s, PDF_s, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))

    # plot lines
    for i in range(distr_num):
        ax.plot([mu_s[i][0], mu_s[i][0]], [mu_s[i][1], mu_s[i][1]],zs=[0,np.max(PDF_s)*1.3], color='black', linestyle='--', linewidth=2)

    ax.set_xlabel(f'$X$', fontsize=20)
    ax.set_ylabel(f'$Y$', fontsize=20)
    ax.set_zlabel(f'$Pdf$', fontsize=20)

    if path_to_save is not None:
        plt.savefig(path_to_save + '/' + img_name + '.jpg')

    #ax.view_init(azim=45)
    plt.show()
