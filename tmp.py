"""
@Author: Du Yunhao
@FileName: tmp.py
@Contact: dyh_bupt@163.com
@Time: 2022/7/17 15:43
@Description: None
"""
import torch
import numpy as np
from pprint import pprint


def ckpt():
    ckpt = torch.load('/data1/dyh/models/mmtracking/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth')
    # ckpt = torch.load('/data1/dyh/models/mmtracking/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth')
    print(ckpt['state_dict'].keys())
    ckpt_ = {
        'meta': ckpt['meta'],
        'state_dict': dict()
    }
    for k, v in ckpt['state_dict'].items():
        if 'detector.' in k:
            k_ = k.replace('detector.', '')
            ckpt_['state_dict'][k_] = v
    torch.save(ckpt_, '/data1/dyh/models/mmtracking/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0_detector.pth')


if __name__ == '__main__':
    a = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    a = a[None]
    print(a)