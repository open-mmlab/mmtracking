# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os.path as osp

import mmcv

try:
    import xlrd
except ImportError:
    xlrd = None
try:
    import xlutils
    from xlutils.copy import copy
except ImportError:
    xlutils = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models metric')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'txt_path', type=str, help='txt path output by benchmark_filter')
    parser.add_argument(
        '--excel', type=str, help='input path of excel to be recorded')
    parser.add_argument(
        '--ncol', type=int, help='Number of column to be modified or appended')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.excel:
        assert args.ncol, 'Please specify "--excel" and "--ncol" ' \
                          'at the same time'
        if xlrd is None:
            raise RuntimeError(
                'xlrd is not installed,'
                'Please use “pip install xlrd==1.2.0” to install')
        if xlutils is None:
            raise RuntimeError(
                'xlutils is not installed,'
                'Please use “pip install xlutils==2.0.0” to install')
        readbook = xlrd.open_workbook(args.excel)

    root_path = args.root
    all_results_dict = {}
    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        model_cfgs = [_ for _ in model_cfgs if 'configs' in _]
        for i, config in enumerate(model_cfgs):
            config = config.strip()
            if len(config) == 0:
                continue

            config_name = osp.split(config)[-1]
            config_name = osp.splitext(config_name)[0]
            result_path = osp.join(root_path, config_name)
            if osp.exists(result_path):
                # 1 read config and excel
                cfg = mmcv.Config.fromfile(config)
                total_epochs = cfg.total_epochs

                # the first metric will be used to find the best ckpt
                has_final_ckpt = True
                if 'vid' in config:
                    eval_metrics = ['bbox_mAP_50']
                elif 'mot' in config:
                    eval_metrics = ['MOTA', 'IDF1']
                    # tracktor and deepsort don't have ckpt.
                    has_final_ckpt = False
                elif 'sot' in config:
                    eval_metrics = ['success', 'norm_precision', 'precision']
                else:
                    raise NotImplementedError(
                        f'Not supported config: {config}')

                if args.excel:
                    xlrw = copy(readbook)
                    if 'vid' in config:
                        sheet = readbook.sheet_by_name('vid')
                        table = xlrw.get_sheet('vid')
                    elif 'mot' in config:
                        sheet = readbook.sheet_by_name('mot')
                        table = xlrw.get_sheet('mot')
                    elif 'sot' in config:
                        sheet = readbook.sheet_by_name('sot')
                        table = xlrw.get_sheet('sot')
                    sheet_info = {}
                    for i in range(6, sheet.nrows):
                        sheet_info[sheet.row_values(i)[0]] = i

                # 2 determine whether total_epochs ckpt exists
                ckpt_path = f'epoch_{total_epochs}.pth'
                if osp.exists(osp.join(result_path, ckpt_path)) or \
                        not has_final_ckpt:
                    log_json_path = list(
                        sorted(glob.glob(osp.join(result_path,
                                                  '*.log.json'))))[-1]

                    # 3 read metric
                    result_dict = dict()
                    with open(log_json_path, 'r') as f:
                        for line in f.readlines():
                            log_line = json.loads(line)
                            if 'mode' not in log_line.keys():
                                continue

                            if log_line['mode'] == 'val' or \
                                    log_line['mode'] == 'test':
                                result_dict[f"epoch_{log_line['epoch']}"] = {
                                    key: log_line[key]
                                    for key in eval_metrics if key in log_line
                                }
                    # 4 find the best ckpt
                    best_epoch_results = dict()
                    for epoch in result_dict:
                        if len(best_epoch_results) == 0:
                            best_epoch_results = result_dict[epoch]
                        else:
                            if best_epoch_results[eval_metrics[
                                    0]] < result_dict[epoch][eval_metrics[0]]:
                                best_epoch_results = result_dict[epoch]

                    for metric in best_epoch_results:
                        if 'success' in best_epoch_results:
                            performance = round(best_epoch_results[metric], 1)
                        else:
                            performance = round(
                                best_epoch_results[metric] * 100, 1)
                        best_epoch_results[metric] = performance
                    all_results_dict[config] = best_epoch_results

                    # update and append excel content
                    if args.excel:
                        performance = ''
                        for metric in best_epoch_results:
                            performance += f'{best_epoch_results[metric]}/'

                        row_num = sheet_info.get(config, None)
                        if row_num:
                            table.write(row_num, args.ncol, performance)
                        else:
                            table.write(sheet.nrows, 0, config)
                            table.write(sheet.nrows, args.ncol, performance)
                        filename, sufflx = osp.splitext(args.excel)
                        xlrw.save(f'{filename}_o{sufflx}')
                        readbook = xlrd.open_workbook(f'{filename}_o{sufflx}')

                else:
                    print(f'{config} not exist: {ckpt_path}')
            else:
                print(f'not exist: {config}')

        # 4 save or print results
        print('===================================')
        for config_name, metrics in all_results_dict.items():
            print(config_name, metrics)
        print('===================================')
        if args.excel:
            print(f'>>> Output {filename}_o{sufflx}')
