# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmengine
import numpy as np

from mmtrack.utils import (plot_norm_precision_curve, plot_precision_curve,
                           plot_success_curve)


def main():
    parser = argparse.ArgumentParser(description='sot plot')
    parser.add_argument(
        'sot_eval_res',
        help='the json/yaml/pickle file path of evaluation results. The '
        'content of the file must be a collection of name/value pairs. The '
        'name is a tracker name. The value is also a collection of name/value '
        'pairs in the format dict(success=np.ndarray, '
        'norm_precision=np.ndarray, precision=np.ndarray)')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show the plotting results')
    parser.add_argument(
        '--plot_save_path',
        default=None,
        type=str,
        help='The saved path of the figure.')
    args = parser.parse_args()

    assert osp.isfile(
        args.sot_eval_res), f'The file {args.sot_eval_res} does not exist'

    all_eval_results = mmengine.load(args.sot_eval_res)
    assert isinstance(all_eval_results, dict)

    tracker_names = []
    all_success = []
    all_norm_precision = []
    all_precision = []
    for tracker_name, scores in all_eval_results.items():
        tracker_names.append(tracker_name)
        if 'success' in scores:
            all_success.append(scores['success'])
        if 'precision' in scores:
            all_precision.append(scores['precision'])
        if 'norm_precision' in scores:
            all_norm_precision.append(scores['norm_precision'])

    if len(all_success) > 0:
        all_success = np.stack(all_success)
        plot_success_curve(
            all_success,
            tracker_names=tracker_names,
            plot_save_path=args.plot_save_path,
            show=args.show)
    if len(all_precision) > 0:
        all_precision = np.stack(all_precision)
        plot_precision_curve(
            all_precision,
            tracker_names=tracker_names,
            plot_save_path=args.plot_save_path,
            show=args.show)
    if len(all_norm_precision) > 0:
        all_norm_precision = np.stack(all_norm_precision)
        plot_norm_precision_curve(
            all_norm_precision,
            tracker_names=tracker_names,
            plot_save_path=args.plot_save_path,
            show=args.show)


if __name__ == '__main__':
    main()
