# Sequence Level Semantics Aggregation for Video Object Detection

## Introduction

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
|    R-50-DC5     |  pytorch  |   7e    | -        | -              | 78.4 | [config](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py) | [model](MODEL_LINK) &#124; [log](LOG_LINK) |
|    R-101-DC5     |  pytorch  |   7e    | -        | -              | 81.5 | [config](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py) | [model](MODEL_LINK) &#124; [log](LOG_LINK) |
