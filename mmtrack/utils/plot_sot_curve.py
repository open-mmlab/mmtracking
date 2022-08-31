# Copyright (c) OpenMMLab. All rights reserved.
# The codes are modified from https://github.com/visionml/pytracking/blob/master/pytracking/analysis/plot_results.py # noqa: E501

from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mmengine.utils import mkdir_or_exist

PALETTE = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
           (0.0, 1.0, 1.0), (0.5, 0.5, 0.5),
           (136.0 / 255.0, 0.0, 21.0 / 255.0),
           (1.0, 127.0 / 255.0, 39.0 / 255.0),
           (0.0, 162.0 / 255.0, 232.0 / 255.0),
           (0.0, 0.5, 0.0), (1.0, 0.5, 0.2), (0.1, 0.4, 0.0), (0.6, 0.3, 0.9),
           (0.4, 0.7, 0.1), (0.2, 0.1, 0.7), (0.7, 0.6, 0.2),
           (1.0, 102.0 / 255.0, 102.0 / 255.0),
           (153.0 / 255.0, 1.0, 153.0 / 255.0),
           (102.0 / 255.0, 102.0 / 255.0, 1.0),
           (1.0, 192.0 / 255.0, 203.0 / 255.0)]
LINE_STYLE = ['-'] * len(PALETTE)


def plot_sot_curve(y: np.ndarray,
                   x: np.ndarray,
                   scores: np.ndarray,
                   tracker_names: List,
                   plot_opts: dict,
                   plot_save_path: Optional[str] = None,
                   show=False):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = np.argsort(scores)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(
            x.tolist(),
            y[id_sort, :].tolist(),
            linewidth=line_width,
            color=PALETTE[len(index_sort) - id - 1],
            linestyle=LINE_STYLE[len(index_sort) - id - 1])

        plotted_lines.append(line[0])

        legend_text.append('{} [{:.1f}]'.format(tracker_names[id_sort],
                                                scores[id_sort]))

    ax.legend(
        plotted_lines[::-1],
        legend_text[::-1],
        loc=legend_loc,
        fancybox=False,
        edgecolor='black',
        fontsize=font_size_legend,
        framealpha=1.0)

    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    if plot_save_path is not None:
        mkdir_or_exist(plot_save_path)
        fig.savefig(
            '{}/{}_plot.pdf'.format(plot_save_path, plot_type),
            dpi=300,
            format='pdf',
            transparent=True)
    plt.draw()
    if show:
        plt.show()


def plot_success_curve(scores: np.ndarray,
                       tracker_names: List,
                       plot_opts: Optional[dict] = None,
                       plot_save_path: Optional[str] = None,
                       show=False):
    assert len(tracker_names) == len(scores)
    success_plot_opts = {
        'plot_type': 'success',
        'legend_loc': 'lower left',
        'xlabel': 'Overlap threshold',
        'ylabel': 'Overlap Precision [%]',
        'xlim': (0, 1.0),
        'ylim': (0, 100),
        'title': 'Success plot'
    }
    if plot_opts is not None:
        success_plot_opts.update(success_plot_opts)
    success = np.mean(scores, axis=1)

    plot_sot_curve(scores, np.arange(0, 1.05, 0.05), success, tracker_names,
                   success_plot_opts, plot_save_path, show)


def plot_norm_precision_curve(scores: np.ndarray,
                              tracker_names: List,
                              plot_opts: Optional[dict] = None,
                              plot_save_path: Optional[str] = None,
                              show=False):
    assert len(tracker_names) == len(scores)
    norm_precision_plot_opts = {
        'plot_type': 'norm_precision',
        'legend_loc': 'lower right',
        'xlabel': 'Location error threshold',
        'ylabel': 'Distance Precision [%]',
        'xlim': (0, 0.5),
        'ylim': (0, 100),
        'title': 'Normalized Precision plot'
    }
    if plot_opts is not None:
        norm_precision_plot_opts.update(norm_precision_plot_opts)

    plot_sot_curve(scores, np.arange(0, 0.51, 0.01), scores[:,
                                                            20], tracker_names,
                   norm_precision_plot_opts, plot_save_path, show)


def plot_precision_curve(scores: np.ndarray,
                         tracker_names: List,
                         plot_opts: Optional[dict] = None,
                         plot_save_path: Optional[str] = None,
                         show=False):
    assert len(tracker_names) == len(scores)
    precision_plot_opts = {
        'plot_type': 'precision',
        'legend_loc': 'lower right',
        'xlabel': 'Location error threshold [pixels]',
        'ylabel': 'Distance Precision [%]',
        'xlim': (0, 50),
        'ylim': (0, 100),
        'title': 'Precision plot'
    }
    if plot_opts is not None:
        precision_plot_opts.update(plot_opts)

    plot_sot_curve(scores, np.arange(0, 51, 1), scores[:, 20], tracker_names,
                   precision_plot_opts, plot_save_path, show)
