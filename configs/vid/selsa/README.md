# Sequence Level Semantics Aggregation for Video Object Detection

## Introduction

[ALGORITHM]

```latex
@inproceedings{wu2019sequence,
  title={Sequence level semantics aggregation for video object detection},
  author={Wu, Haiping and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9217--9225},
  year={2019}
}
```

## Results and models on ImageNet VID dataset

We observe around 1 mAP fluctuations in performance, and provide the best model.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  pytorch  |   7e    | 3.49        | 7.5            | 78.4 | [config](selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835.log.json) |
|    R-101-DC5     |  pytorch  |   7e    | 5.18        | 7.2              | 81.5 | [config](selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724.log.json) |
|    X-101-DC5     |  pytorch  |   7e    | 9.15        | -              | 83.1 | [config](selsa_faster_rcnn_x101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_x101_dc5_1x_imagenetvid/selsa_faster_rcnn_x101_dc5_1x_imagenetvid_20210825_205641-10252965.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_x101_dc5_1x_imagenetvid/selsa_faster_rcnn_x101_dc5_1x_imagenetvid_20210825_205641.log.json) |
