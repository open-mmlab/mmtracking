# Flow-guided Feature Aggregation for Video Object Detection

## Introduction

[ALGORITHM]

```latex
@inproceedings{zhu2017flow,
  title={Flow-guided feature aggregation for video object detection},
  author={Zhu, Xizhou and Wang, Yujie and Dai, Jifeng and Yuan, Lu and Wei, Yichen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={408--417},
  year={2017}
}
```

## Results and models on ImageNet VID dataset

We observe around 1 mAP fluctuations in performance, and provide the best model.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  pytorch  |   7e    | 4.10        | 6.9              | 74.7 | [config](fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid_20201228_022657-f42016f3.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid_20201228_022657.log.json) |
|    R-101-DC5     |  pytorch  |   7e    | 5.80        | 6.4              | 77.8 | [config](fgfa_faster_rcnn_r101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid_20201219_011831-9c9d8183.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid_20201219_011831.log.json) |
|    X-101-DC5     |  pytorch  |   7e    | 9.74        | -              | 79.6 | [config](fgfa_faster_rcnn_x101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid_20210818_223334-8723c594.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid_20210818_223334.log.json) |
