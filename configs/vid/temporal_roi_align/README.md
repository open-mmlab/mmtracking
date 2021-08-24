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

We observe around 0.5 mAP fluctuations in performance, and provide the best model.

We find that using 3 selsa modules can bring slightly gains (around 0.2 mAP) than using 2 selsa modules in `SELSA + TRoI` method, and the performance will drop when changing the number of selsa modules in `SELSA` method from 2 to 3. Therefore, we adopt using 3 selsa modules in `SELSA + TRoI` method, meanwhile keeping using 2 selsa modules in `SELSA` method.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  pytorch  |   7e    | 4.14        | -            | 79.8 | [config](selsa_troi_faster_rcnn_r50_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troi_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troi_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troi_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troi_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714.log.json) |
|    R-101-DC5     |  pytorch  |   7e    | 5.83        | -              | 82.6 | [config](selsa_troi_faster_rcnn_r101_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troi_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troi_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621-22cb96b9.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troi_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troi_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621.log.json) |
