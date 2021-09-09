# Temporal RoI Align for Video Object Recognition

## Introduction

[ALGORITHM]

```latex
@inproceedings{gong2021temporal,
  title={Temporal ROI Align for Video Object Recognition},
  author={Gong, Tao and Chen, Kai and Wang, Xinjiang and Chu, Qi and Zhu, Feng and Lin, Dahua and Yu, Nenghai and Feng, Huamin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={2},
  pages={1442--1450},
  year={2021}
}
```

## Results and models on ImageNet VID dataset

We observed that the performance of this method has a fluctuation of about 0.5 mAP. The checkpoint provided below is the best one from two experiments.

Note that the numbers of selsa modules in this method and `SELSA` are 3 and 2 respectively. This is because another selsa modules improve this method by 0.2 points but degrade `SELSA` by 0.5 points. We choose the best settings for the two methods for a fair comparison.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  pytorch  |   7e    | 4.14        | -            | 79.8 | [config](selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714.log.json) |
|    R-101-DC5     |  pytorch  |   7e    | 5.83        | -              | 82.6 | [config](selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621-22cb96b9.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621.log.json) |
|    X-101-DC5     |  pytorch  |   7e    | 9.74        | -              | 84.1 | [config](selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036-4471ac42.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036.log.json) |
