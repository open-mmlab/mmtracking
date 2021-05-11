# Tracking without Bells and Whistles

## Introduction

[ALGORITHM]

```latex
@inproceedings{bergmann2019tracking,
  title={Tracking without bells and whistles},
  author={Bergmann, Philipp and Meinhardt, Tim and Leal-Taixe, Laura},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={941--951},
  year={2019}
}
```

## Results and models on MOT15

We implement Tracktor with independent detector and ReID models. To train a model by yourself, you need to train a detector following [here](../../det/) and also train a ReID model.
The configs in this folder are basiclly for inference.

Currently we do not support training ReID models.
For MOT15, MOT16, and MOT20, we train the ReID model following by [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw).
For MOT17, we directly use the ReID model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). These missed features will be supported in the future.

The implementations of Tracktor follow the offical practices.

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   | 62.4 | 63.6 | 1323 | 6642 | 123 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot15-f7980743.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   | 65.9 | 66.3 | 3404 | 3746 | 184 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot15-f7980743.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     |   | 70.4 | 64.2 | 1125 | 11355 | 274 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot15-f7980743.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     |   | 80.0 | 68.1 | 3882 | 4255 | 478 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot15-f7980743.pth) |

## Results and models on MOT16

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   | 53.8 | 59.4 | 459 | 24007 | 186 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half-4c1b09ac.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot16-a2e459b3.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   | 62.9 | 64.1 | 4389 | 14905 | 817 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half-4c1b09ac.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot16-a2e459b3.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     |   | 61.5 | 62.4 | 1162 | 40896 | 423 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-ccb2ff52.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot16-a2e459b3.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     |   | 74.9 | 65.7 | 7767 | 18517 | 1469 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-ccb2ff52.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot16-a2e459b3.pth) |

## Results and models on MOT17

In this table below, the result marked with * (the last line) is the offical one.
Our implementation outperform it by 4.9 points on MOTA and 3.3 points on IDF1.

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     | 3.2  | 57.3 | 63.4 | 1254 | 67091 | 613 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     | 3.1  | 64.1 | 66.5 | 11088 | 45762 | 1224 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     | 3.2  | 69.3 | 69.3 | 4010 | 97918 | 1527 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     | 3.1  | 82.1 | 73.4 | 12789 | 44631 | 2988 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | train      | test     | Y     |   | 61.2 | 58.4 | 8612 | 207628 | 2637 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN* | R50 | train     | test     | Y     | -    | 56.3 | 55.1 | 8866 | 235449 | 1987 | -    | -     |

## Results and models on MOT20

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   | 70.7 | 65.5 | 3513 | 175352 | 1437 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half-860a6c6f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot20-367af9dd.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   | 71.0 | 63.6 | 5414 | 171493 | 1611 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half-860a6c6f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot20-367af9dd.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     |   | 76.9 | 69.3 | 1181 | 258494 | 2123 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-ef875499.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot20-367af9dd.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     |   | 77.2 | 68.8 | 1250 | 255362 | 2291 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-ef875499.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot20-367af9dd.pth) |
