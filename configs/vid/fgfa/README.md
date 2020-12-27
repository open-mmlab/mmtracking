# Flow-guided Feature Aggregation for Video Object Detection

## Introduction

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
|    R-50-DC5     |  pytorch  |   7e    | -        | -              | ??? | [config](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/fgfa/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py) | [model](MODEL_LINK) &#124; [log](LOG_LINK) |
|    R-101-DC5     |  pytorch  |   7e    | -        | -              | 77.6? | [config](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid.py) | [model](MODEL_LINK) &#124; [log](LOG_LINK) |
