# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    exp_dir = osp.dirname(in_file)
    log_json_path = list(sorted(glob.glob(osp.join(exp_dir,
                                                   '*.log.json'))))[-1]
    model_time = osp.split(log_json_path)[-1].split('.')[0]

    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'_{model_time}' + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])
    cp_log_json_path = out_file_name + f'_{osp.basename(log_json_path)}'
    subprocess.Popen(['cp', log_json_path, cp_log_json_path])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
