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

We implement Tracktor with independent detector and ReID models. To train a model by yourself, you need to train a detector following [here](../../det/) and also train a ReID model following [here](../../reid/).
The configs in this folder are basiclly for inference.

## Results and models on MOT15

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |      | 62.4 | 63.3 | 1323 | 6642 | 125 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15-30ba14d3.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |      | 65.9 | 66.7 | 3404 | 3746 | 188 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15-30ba14d3.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     |      | 70.4 | 64.4 | 1125 | 11355 | 277 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15-30ba14d3.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     |      | 80.0 | 68.8 | 3883 | 4256 | 473 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15-30ba14d3.pth) |

## Results and models on MOT16

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |      | 53.8 | 59.3 | 459 | 24007 | 179 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half-4c1b09ac.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16-244ecae5.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |      | 63.1 | 64.8 | 4389 | 14905 | 383 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half-4c1b09ac.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16-244ecae5.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     |      | 61.5 | 62.6 | 1162 | 30896 | 403 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-ccb2ff52.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16-244ecae5.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     |      | 75.1 | 67.2 | 7766 | 18516 | 1220 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot16-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-ccb2ff52.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16-244ecae5.pth) |

## Results and models on MOT17

The implementations of Tracktor follow the offical practices.
In the table below, the result marked with * (the last line) is the offical one.
Our implementation outperform it by 4.9 points on MOTA and 3.3 points on IDF1.

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     | 3.2  | 57.3 | 63.4 | 1254 | 67091 | 614 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     | 3.1  | 64.1 | 66.9 | 11088 | 45762 | 1233 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     | 3.2  | 69.3 | 69.4 | 4010 | 97918 | 1540 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     | 3.1  | 82.1 | 73.2 | 12795 | 44637 | 3033 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) |
| R50-FasterRCNN-FPN | R50 | train      | test     | Y     | 3.2  | 61.2 | 58.4 | 8609 | 207627 | 2634 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) |
| R50-FasterRCNN-FPN* | R50 | train     | test     | Y     | -    | 56.3 | 55.1 | 8866 | 235449 | 1987 | -    | -     |

## Results and models on MOT20

The implementations of Tracktor follow the offical practices.
In the table below, the result marked with * (the last line) is the offical one.
Our implementation outperform it by 5.0 points on MOTA and 2.3 points on IDF1.

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |      | 70.7 | 65.5 | 3513 | 175352 | 1436 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half-860a6c6f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20-afbdfea4.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |      | 71.0 | 63.6 | 5414 | 171493 | 1611 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half-860a6c6f.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20-afbdfea4.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | Y     |      | 76.9 | 69.3 | 1181 | 258494 | 2119 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-ef875499.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20-afbdfea4.pth) |
| R50-FasterRCNN-FPN | R50 | train      | train    | N     |      | 77.2 | 68.8 | 1250 | 255362 | 2285 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-ef875499.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20-afbdfea4.pth) |
| R50-FasterRCNN-FPN | R50 | train      | test     | Y     |      | 57.6 | 55.0 | 7973 | 209676 | 1796 |  [config](tracktor_faster-rcnn_r50_fpn_4e_mot20-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-ef875499.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20-afbdfea4.pth) |
| R50-FasterRCNN-FPN* | R50 | train     | test     | Y     | -    | 52.6 | 52.7 | 6930 | 236680 | 1648 | -    | -     |
