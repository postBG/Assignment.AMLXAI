import os
import re
from itertools import groupby

import numpy as np

import matplotlib.pyplot as plt


class ResultData(object):
    def __init__(self, results, lengths):
        assert len(results) == len(lengths)
        self.results = results
        self.lengths = lengths

    def averages(self, percent=False):
        avgs = [sum(result) / l for result, l in zip(self.results, self.lengths)]
        return [avg * 100 for avg in avgs] if percent else avgs

    def average(self, idx, percent=False):
        avg = sum(self.results[idx]) / self.lengths[idx]
        return avg * 100 if percent else avg


def average_result_data_list(result_data_list, percent=True):
    averages = np.array([result_data.averages(percent=percent) for result_data in result_data_list])
    return list(np.average(averages, axis=0))


def read_result(path):
    rel_path = os.path.join('result_data', path)
    with open(rel_path, 'r') as f:
        lines = f.readlines()
        results = [[float(x) for x in l.strip().split()] for l in lines]
        lengths = list(range(1, len(lines) + 1))
        return ResultData(results, lengths)


def retrieve_reg_from_path(path):
    return re.findall('(CIFAR100|MNIST)_([\w\d]+?)_', path)[0][1]


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


def plot_search_graphs_from_result_paths(paths, save_path='', ylim=None):
    results = [read_result(path) for path in paths]
    plot_data = [result.averages(percent=True) for result in results]
    lambdas = [float(re.findall('lamb_([\d.]+)', path)[0]) for path in paths]
    palette = ['000000', '55415f', '646964', 'd77355', '508cd7', '64b964', 'e6c86e', 'dcf5ff']

    plot_hyperparameters_search_graph(lambdas, palette, plot_data, save_path, ylim)


def plot_comparing_graphs_from_result_paths(paths, save_path='', ylim=None):
    reg_result_pairs = [(retrieve_reg_from_path(path), read_result(path)) for path in paths]
    reg_result_pairs = sorted(reg_result_pairs, key=lambda x: x[0])
    groups = groupby(reg_result_pairs, key=lambda x: x[0])
    groups = [(reg, list(result_data for _, result_data in pairs)) for reg, pairs in groups]
    averages = [(reg, average_result_data_list(result_data_list)) for reg, result_data_list in groups]
    regularizers, plot_data = list(map(list, (zip(*averages))))

    palette = ['ef233c', '008000', '0a85ed']

    plot_comparing_graph(palette, plot_data, regularizers, save_path, ylim)
