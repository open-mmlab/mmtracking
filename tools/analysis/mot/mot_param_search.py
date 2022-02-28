# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from itertools import product

import mmcv
import torch
from dotty_dict import dotty
from mmcv import Config, DictAction, get_logger, print_log
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import build_dataset

from mmtrack.models import build_tracker


def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--log', help='log file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
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


def get_search_params(cfg, search_params=None, prefix=None, logger=None):
    if search_params is None:
        search_params = dict()
    for k, v in cfg.items():
        if prefix is not None:
            entire_k = prefix + '.' + k
        else:
            entire_k = k
        if isinstance(v, list):
            print_log(f'search `{entire_k}` in {v}.', logger)
            search_params[entire_k] = v
        if isinstance(v, dict):
            search_params = get_search_params(v, search_params, entire_k,
                                              logger)
    return search_params


def main():

    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.datasets import build_dataloader
        from mmdet.models import build_detector as build_model
    else:
        from mmtrack.apis import multi_gpu_test, single_gpu_test
        from mmtrack.datasets import build_dataloader
        from mmtrack.models import build_model
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # cfg.model.pretrains = None
    if hasattr(cfg.model, 'detector'):
        cfg.model.detector.pretrained = None
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

    logger = get_logger('ParamsSearcher', log_file=args.log)
    # get all cases
    search_params = get_search_params(cfg.model.tracker, logger=logger)
    combinations = [p for p in product(*search_params.values())]
    search_cfgs = []
    for c in combinations:
        search_cfg = dotty(cfg.model.tracker.copy())
        for i, k in enumerate(search_params.keys()):
            search_cfg[k] = c[i]
        search_cfgs.append(dict(search_cfg))
    print_log(f'Totally {len(search_cfgs)} cases.', logger)
    # init with the first one
    cfg.model.tracker = search_cfgs[0].copy()

    # build the model and load checkpoint
    if cfg.get('test_cfg', False):
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)
    # We need call `init_weights()` to load pretained weights in MOT task.
    model.init_weights()
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

    print_log(f'Record {cfg.search_metrics}.', logger)
    for i, search_cfg in enumerate(search_cfgs):
        if not distributed:
            model.module.tracker = build_tracker(search_cfg)
            outputs = single_gpu_test(model, data_loader, args.show,
                                      args.show_dir)
        else:
            model.module.tracker = build_tracker(search_cfg)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                results = dataset.evaluate(outputs, **eval_kwargs)
                _records = []
                for k in cfg.search_metrics:
                    if isinstance(results[k], float):
                        _records.append(f'{(results[k]):.3f}')
                    else:
                        _records.append(f'{(results[k])}')
                print_log(f'{combinations[i]}: {_records}', logger)


if __name__ == '__main__':
    main()
