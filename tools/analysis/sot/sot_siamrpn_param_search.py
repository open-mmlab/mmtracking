# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import numpy as np
import torch
from mmcv import Config, DictAction, get_logger, print_log
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import build_dataset


def parse_range(range_str):
    range_list = range_str.split(',')
    assert len(range_list) == 3 and float(range_list[1]) >= float(
        range_list[0])
    param = map(float, range_list)
    return np.round(np.arange(*param), decimals=2)


def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
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
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--log', help='log file', default=None)
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
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


def main():
    args = parse_args()

    assert args.eval or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (eval/show the '
         'results) with the argument "--eval"'
         ', "--show" or "--show-dir"')

    cfg = Config.fromfile(args.config)

    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.datasets import build_dataloader
        from mmdet.models import build_detector as build_model
        if 'detector' in cfg.model:
            cfg.model = cfg.model.detector
    elif cfg.get('USE_MMCLS', False):
        from mmtrack.apis import multi_gpu_test, single_gpu_test
        from mmtrack.datasets import build_dataloader
        from mmtrack.models import build_reid as build_model
        if 'reid' in cfg.model:
            cfg.model = cfg.model.reid
    else:
        from mmtrack.apis import multi_gpu_test, single_gpu_test
        from mmtrack.datasets import build_dataloader
        from mmtrack.models import build_model
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    logger = get_logger('SOTParamsSearcher', log_file=args.log)

    # build the model and load checkpoint
    if cfg.get('test_cfg', False):
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    if not hasattr(model, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    # init best_score, best_results and best parames
    if 'meta' in checkpoint and 'hook_msgs' in checkpoint[
            'meta'] and 'best_score' in checkpoint['meta']['hook_msgs']:
        best_score = checkpoint['meta']['hook_msgs']['best_score']
    else:
        best_score = 0

    key_metric = cfg.evaluation.save_best
    best_result = {f'{key_metric}': best_score}

    best_params = dict(
        penalty_k=cfg.model.test_cfg.rpn.penalty_k,
        lr=cfg.model.test_cfg.rpn.lr,
        win_influ=cfg.model.test_cfg.rpn.window_influence)
    print_log(f'init best score as: {best_score}', logger)
    print_log(f'init best params as: {best_params}', logger)

    num_cases = len(args.penalty_k_range) * len(args.lr_range) * len(
        args.win_influ_range)
    case_count = 0

    # compare function setting in parameter search
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    compare_func = rule_map[cfg.evaluation.rule]

    for penalty_k in args.penalty_k_range:
        for lr in args.lr_range:
            for win_influ in args.win_influ_range:
                case_count += 1
                cfg.model.test_cfg.rpn.penalty_k = penalty_k
                cfg.model.test_cfg.rpn.lr = lr
                cfg.model.test_cfg.rpn.window_influence = win_influ
                print_log(f'-----------[{case_count}/{num_cases}]-----------',
                          logger)
                print_log(
                    f'penalty_k={penalty_k} lr={lr} win_influence={win_influ}',
                    logger)

                if not distributed:
                    outputs = single_gpu_test(
                        model,
                        data_loader,
                        args.show,
                        args.show_dir,
                        show_score_thr=args.show_score_thr)
                else:
                    outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                             args.gpu_collect)

                rank, _ = get_dist_info()
                if rank == 0:
                    kwargs = args.eval_options if args.eval_options else {}
                    if args.eval:
                        eval_kwargs = cfg.get('evaluation', {}).copy()
                        # hard-code way to remove EvalHook args
                        eval_hook_args = [
                            'interval', 'tmpdir', 'start', 'gpu_collect',
                            'save_best', 'rule', 'by_epoch'
                        ]
                        for key in eval_hook_args:
                            eval_kwargs.pop(key, None)
                        eval_kwargs.update(dict(metric=args.eval, **kwargs))
                        eval_results = dataset.evaluate(outputs, **eval_kwargs)
                        print_log(f'evaluation results: {eval_results}',
                                  logger)
                        print_log('------------------------------------------',
                                  logger)

                        if compare_func(eval_results[key_metric],
                                        best_result[key_metric]):
                            best_result = eval_results
                            best_params['penalty_k'] = penalty_k,
                            best_params['lr'] = lr,
                            best_params['win_influ'] = win_influ

                        print_log(
                            f'The current best evaluation results: \
                                {best_result}', logger)
                        print_log(f'The current best params: {best_params}',
                                  logger)

    print_log(
        f'After parameter searching, the best evaluation results: \
            {best_result}', logger)
    print_log(f'After parameter searching, the best params: {best_params}',
              logger)


if __name__ == '__main__':
    main()
