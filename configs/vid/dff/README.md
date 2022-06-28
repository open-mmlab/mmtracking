# Deep Feature Flow for Video Recognition

## Abstract

<!-- [ABSTRACT] -->

Deep convolutional neutral networks have achieved great success on image recognition tasks. Yet, it is nontrivial to transfer the state-of-the-art image recognition networks to videos as per-frame evaluation is too slow and unaffordable. We present deep feature flow, a fast and accurate framework for video recognition. It runs the expensive convolutional sub-network only on sparse key frames and propagates their deep feature maps to other frames via a flow field. It achieves significant speedup as flow computation is relatively fast. The end-to-end training of the whole architecture significantly boosts the recognition accuracy. Deep feature flow is flexible and general. It is validated on two video datasets on object detection and semantic segmentation. It significantly advances the practice of video recognition tasks.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142985441-53afc070-6646-404b-869a-e967dc92bde6.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

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

| Method | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 |                        Config                        |                                                                                                                                                                   Download                                                                                                                                                                   |
| :----: | :-------: | :-----: | :-----: | :------: | :------------: | :-------: | :--------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  DFF   | R-50-DC5  | pytorch |   7e    |   2.50   |      44.0      |   70.3    | [config](dff_faster_rcnn_r50_dc5_1x_imagenetvid.py)  |   [model](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid/dff_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_213250-548911a4.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid/dff_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_213250.log.json)   |
|  DFF   | R-101-DC5 | pytorch |   7e    |   3.25   |      39.8      |   73.5    | [config](dff_faster_rcnn_r101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720.log.json) |
|  DFF   | X-101-DC5 | pytorch |   7e    |   4.95   |       -        |   75.5    | [config](dff_faster_rcnn_x101_dc5_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_x101_dc5_1x_imagenetvid/dff_faster_rcnn_x101_dc5_1x_imagenetvid_20210819_095932-0a9e6cb5.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_x101_dc5_1x_imagenetvid/dff_faster_rcnn_x101_dc5_1x_imagenetvid_20210819_095932.log.json) |
