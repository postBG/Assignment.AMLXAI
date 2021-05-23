import os
import re
from itertools import groupby

import numpy as np

import matplotlib.pyplot as plt


def read_losses(path):
    rel_path = os.path.join('result_data', 'bests', path, 'loss.txt')
    with open(rel_path, 'r') as f:
        lines = f.readlines()
        results = [float(l) for l in lines]
        return results


def format_plot(plot_data, save_path, ylim):
    task_length = len(plot_data[0])
    xticks = list(range(task_length))
    plt.xticks(xticks, labels=[f't{t + 1}' for t in xticks])
    plt.legend(loc='lower left')
    plt.xlabel('Task')
    plt.ylabel('Acc. (%)')
    if ylim:
        plt.ylim(ylim)
    if save_path:
        extension = save_path.split('.')[-1]
        plt.savefig(save_path, format=extension)


def plot_hyperparameters_search_graph(lambdas, palette, plot_data, save_path, ylim=None):
    linewidth = 1.5
    plt.rcParams['figure.figsize'] = 12, 5
    for i, data in enumerate(plot_data):
        plt.plot(data, linewidth=linewidth, color=f'#{palette[i]}',
                 label=r'$\lambda = {}$'.format(lambdas[i]))
    format_plot(plot_data, save_path, ylim)


def plot_comparing_graph(palette, plot_data, regularizers, save_path, ylim):
    linewidth = 1.5
    plt.rcParams['figure.figsize'] = 12, 5
    for i, data in enumerate(plot_data):
        plt.plot(data, linewidth=linewidth, color=f'#{palette[i]}',
                 label=f'{regularizers[i]}')
    format_plot(plot_data, save_path, ylim)


def plot_figure2(paths, save_path='', ylim=None):
    results = [read_losses(path) for path in paths]
    names = ['FOMAML', 'REPTILE']
    palette = ['ef233c', '508cd7']
    linewidth = 1.5
    plt.rcParams['figure.figsize'] = 12, 5
    for i, data in enumerate(results):
        plt.plot(data, linewidth=linewidth, color=f'#{palette[i]}', label=f'{names[i]}')

    num_updates = len(results[0])
    xticks = list(range(num_updates))
    plt.xticks(xticks, labels=[f'{t}' for t in xticks])
    plt.legend(loc='upper right')
    plt.xlabel('Number of Gradient Steps')
    plt.ylabel('CE loss')
    if ylim:
        plt.ylim(ylim)
    if save_path:
        extension = save_path.split('.')[-1]
        plt.savefig(save_path, format=extension)


def plot_figure2_sin(paths, save_path='', ylim=None):
    results = [read_losses(path) for path in paths]
    names = ['MAML', 'FOMAML', 'REPTILE']
    palette = ['d77355', 'ef233c', '508cd7']
    linewidth = 1.5
    plt.rcParams['figure.figsize'] = 12, 5
    for i, data in enumerate(results):
        plt.plot(data, linewidth=linewidth, color=f'#{palette[i]}', label=f'{names[i]}')

    num_updates = len(results[0])
    xticks = list(range(num_updates))
    plt.xticks(xticks, labels=[f'{t}' for t in xticks])
    plt.legend(loc='upper right')
    plt.xlabel('Number of Gradient Steps')
    plt.ylabel('MSE loss')
    if ylim:
        plt.ylim(ylim)
    if save_path:
        extension = save_path.split('.')[-1]
        plt.savefig(save_path, format=extension)
