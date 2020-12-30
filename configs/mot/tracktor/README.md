# Tracking without Bells and Whistles

## Introduction

```latex
@inproceedings{bergmann2019tracking,
  title={Tracking without bells and whistles},
  author={Bergmann, Philipp and Meinhardt, Tim and Leal-Taixe, Laura},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={941--951},
  year={2019}
}
```

## Results and models on MOT17

We implement Tracktor following the offical practices.

We train the detector with MMDetection. However, currently we do not support training ReID models.
We directly use the model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw) for inference. These missed features will be supported in the future.

The results marked with * (the last line) is the offical implementations. Our implementations outperform it by 4.9 points on MOTA and 3.3 points on IDF1.

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     | 3.2  | 57.3 | 63.4 | 1254 | 67091 | 613 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/v0.5/mot/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/v0.5/mot/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     | 3.1  | 64.1 | 66.5 | 11088 | 45762 | 1224 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/v0.5/mot/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/v0.5/mot/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     | 3.2  | 69.3 | 69.3 | 4010 | 97918 | 1527 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/v0.5/mot/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/v0.5/mot/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     | 3.1  | 82.1 | 73.4 | 12789 | 44631 | 2988 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py) | [detector](https://download.openmmlab.com/mmtracking/v0.5/mot/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/v0.5/mot/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | train      | test     | Y     | 3.2  | 61.2 | 58.4 | 8612 | 207628 | 2637 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/v0.5/mot/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/v0.5/mot/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN* | R50 | train     | test     | Y     | -    | 56.3 | 55.1 | 8866 | 235449 | 1987 | -    | -     |
