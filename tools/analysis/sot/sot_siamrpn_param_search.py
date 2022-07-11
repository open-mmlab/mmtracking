# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import numpy as np
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmtrack.utils import register_all_modules


def parse_range(range_str):
    range_list = range_str.split(',')
    assert len(range_list) == 3 and float(range_list[1]) >= float(
        range_list[0])
    param = map(float, range_list)
    return np.round(np.arange(*param), decimals=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMTrack test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--penalty-k-range',
        type=parse_range,
        help="the range of hyper-parameter 'penalty_k' in SiamRPN++; the format \
            is 'start,stop,step'")
    parser.add_argument(
        '--lr-range',
        type=parse_range,
        help="the range of hyper-parameter 'lr' in SiamRPN++; the format is \
            'start,stop,step'")
    parser.add_argument(
        '--win-influ-range',
        type=parse_range,
        help="the range of hyper-parameter 'window_influence' in SiamRPN++; the \
            format is 'start,stop,step'")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def parameter_search(runner, args):
    cfg = runner.cfg
    logger = runner.logger

    # calculate the number of all search cases and set comparing standard.
    num_cases = len(args.penalty_k_range) * len(args.lr_range) * len(
        args.win_influ_range)
    case_count = 0
    # compare function setting in parameter search. Now, the default comparing
    # ruler is  `greater` because the model doesn't record comparing ruler
    # of metrics in ``MMEngine``.
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    compare_func = rule_map['greater']

    if cfg.test_evaluator.metric == 'OPE':
        eval_metrics = ['success', 'norm_precision', 'precision']
        key_metric = 'success'
    else:
        eval_metrics = ['eao', 'accuracy', 'robustness', 'num_fails']
        key_metric = 'eao'

    checkpoint = runner.load_checkpoint(args.checkpoint)

    # init best_score, best_results and best parames
    if 'meta' in checkpoint and 'hook_msgs' in checkpoint[
            'meta'] and key_metric in checkpoint['meta']['hook_msgs']:
        best_score = checkpoint['meta']['hook_msgs'][key_metric]
    else:
        best_score = 0
    best_result = {f'{key_metric}': best_score}

    best_params = dict(
        penalty_k=cfg.model.test_cfg.rpn.penalty_k,
        lr=cfg.model.test_cfg.rpn.lr,
        win_influ=cfg.model.test_cfg.rpn.window_influence)
    print_log(f'init best score as: {best_score}', logger)
    print_log(f'init best params as: {best_params}', logger)

    for penalty_k in args.penalty_k_range:
        for lr in args.lr_range:
            for win_influ in args.win_influ_range:
                case_count += 1
                runner.model.test_cfg.rpn.penalty_k = penalty_k
                runner.model.test_cfg.rpn.lr = lr
                runner.model.test_cfg.rpn.window_influence = win_influ
                print_log(f'-----------[{case_count}/{num_cases}]-----------',
                          logger)
                print_log(
                    f'penalty_k={penalty_k} lr={lr} win_influence={win_influ}',
                    logger)

                # start testing
                runner.test()

                # parse the eluation results
                res = dict()
                for metric in eval_metrics:
                    res[metric] = runner.message_hub.get_scalar(
                        'test/sot/' + metric).current()

                # show results
                rank, _ = get_dist_info()
                if rank == 0:
                    print_log(f'evaluation results: {res}', logger)
                    print_log('------------------------------------------',
                              logger)
                    if compare_func(res[key_metric], best_result[key_metric]):
                        best_result = res
                        best_params['penalty_k'] = penalty_k
                        best_params['lr'] = lr
                        best_params['win_influ'] = win_influ
                    print_log(
                        f'The current best evaluation results: {best_result}',
                        logger)
                    print_log(f'The current best params: {best_params}',
                              logger)

    print_log(
        'After parameter searching, the best evaluation results: '
        f'{best_result}', logger)
    print_log(f'After parameter searching, the best params: {best_params}',
              logger)


def main():
    args = parse_args()

    # register all modules in mmtrack into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    parameter_search(runner, args)


if __name__ == '__main__':
    main()
