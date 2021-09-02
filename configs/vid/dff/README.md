# Deep Feature Flow for Video Recognition

## Introduction

[ALGORITHM]

```latex
@inproceedings{zhu2017deep,
  title={Deep feature flow for video recognition},
  author={Zhu, Xizhou and Xiong, Yuwen and Dai, Jifeng and Yuan, Lu and Wei, Yichen},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2349--2358},
  year={2017}
}
```

## Results and models on ImageNet VID dataset

We observe around 1 mAP fluctuations in performance, and provide the best model.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  pytorch  |   7e    | 2.50        | 44.0              | 70.3 | [config](dff_faster_rcnn_r50_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid/dff_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_213250-548911a4.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid/dff_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_213250.log.json) |
|    R-101-DC5     |  pytorch  |   7e    | 3.25        | 39.8              | 73.5 | [config](dff_faster_rcnn_r101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720.log.json) |
|    X-101-DC5     |  pytorch  |   7e    | 4.95       | -              | 75.5 | [config](dff_faster_rcnn_x101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_x101_dc5_1x_imagenetvid/dff_faster_rcnn_x101_dc5_1x_imagenetvid_20210819_095932-0a9e6cb5.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_x101_dc5_1x_imagenetvid/dff_faster_rcnn_x101_dc5_1x_imagenetvid_20210819_095932.log.json) |
