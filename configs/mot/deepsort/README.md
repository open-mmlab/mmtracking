# Deep SORT

## Introduction

[ALGORITHM]

```latex
@inproceedings{bewley2016simple,
  title={Simple online and realtime tracking},
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  pages={3464--3468},
  year={2016},
  organization={IEEE}
}
@inproceedings{wojke2017simple,
  title={Simple online and realtime tracking with a deep association metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE international conference on image processing (ICIP)},
  pages={3645--3649},
  year={2017},
  organization={IEEE}
}
```
## Results and models on MOT15
We implement SORT and DeepSORT with independent detector and ReID models. To train a model by yourself, you need to train a detector following [here](../../det/) and also train a ReID model.
The configs in this folder are basiclly for inference.

Currently we do not support training ReID models.
We directly use the ReID model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). These missed features will be supported in the future.

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | -  | half-train | half-val | Y     |  |   45.5 | 40.3 | 707 | 9971 | 1059 | [config](sort_faster-rcnn_fpn_4e_mot15-public-half.py) |  [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | -  | half-train | half-val | N     |  |   62.4 | 59.2 | 4252 | 3138 | 701 | [config](sort_faster-rcnn_fpn_4e_mot15-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   |  49.7 | 49.2 | 704   | 9968 | 157 | [config](deepsort_faster-rcnn_fpn_4e_mot15-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   |  64.1 | 60.8 | 4250 | 3136 | 334 | [config](deepsort_faster-rcnn_fpn_4e_mot15-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | - | train | train | Y               |   | 45.9 | 34.8 | 519 | 20444 | 2375 | [config](sort_faster-rcnn_fpn_4e_mot15-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |
| R50-FasterRCNN-FPN | - | train | train | N               |   | 76.1 | 59.8 | 5797 | 2791 | 1722 | [config](sort_faster-rcnn_fpn_4e_mot15-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |

## Results and models on MOT16
|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | -  | half-train | half-val | Y     |   |   38.1 | 36.8 | 145 | 31173 | 1694 | [config](sort_faster-rcnn_fpn_4e_mot16-public-half.py) |  [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | -  | half-train | half-val | N     |   |  60.4  | 56.8 | 5913 | 13113 | 2120 | [config](sort_faster-rcnn_fpn_4e_mot16-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   |  40.4 | 55.3 | 147   | 31175 | 469 | [config](deepsort_faster-rcnn_fpn_4e_mot16-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   |  61.9 | 71.6 | 5888 | 13088 | 1364 | [config](deepsort_faster-rcnn_fpn_4e_mot16-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | - | train | train | Y               |   | 39.5 | 32.5 | 196 | 62975 | 3639 | [config](sort_faster-rcnn_fpn_4e_mot16-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |
| R50-FasterRCNN-FPN | - | train | train | N               |   | 73.6 | 57.6 | 11036 | 13055 | 5044 | [config](sort_faster-rcnn_fpn_4e_mot16-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |

## Results and models on MOT17

|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | -  | half-train | half-val | Y     | 28.3 |   46.0 | 46.6 | 289 | 82451 | 4581 | [config](sort_faster-rcnn_fpn_4e_mot17-public-half.py) |  [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | -  | half-train | half-val | N     | 18.6 |   62.0 | 57.8 | 15171 | 40437 | 5841 | [config](sort_faster-rcnn_fpn_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     | 20.4  |  48.1 | 60.8 | 283   | 82445 | 1199 | [config](deepsort_faster-rcnn_fpn_4e_mot17-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     | 13.8  |  63.8 | 69.6 | 15060 | 40326 | 3183 | [config](deepsort_faster-rcnn_fpn_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | - | train | train | Y               | 28.3  | 50.9 | 44.5 | 1108 | 153950 | 10522 | [config](sort_faster-rcnn_fpn_4e_mot17-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |
| R50-FasterRCNN-FPN | - | train | train | N               | 18.6  | 80.8 | 61.3 | 21537 | 29280 | 13947 | [config](sort_faster-rcnn_fpn_4e_mot17-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |

## Results and models on MOT20
|    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | -  | half-train | half-val | Y     |  |   44.3 | 26.8 | 345 | 319355 | 23125 | [config](sort_faster-rcnn_fpn_4e_mot20-public-half.py) |  [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | -  | half-train | half-val | N     |  |   62.4 | 59.2 | 4252 | 3138 | 701 | [config](sort_faster-rcnn_fpn_4e_mot20-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   |  49.7 | 49.2 | 704   | 9968 | 157 | [config](deepsort_faster-rcnn_fpn_4e_mot20-public-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   |  64.1 | 60.8 | 4250 | 3136 | 334 | [config](deepsort_faster-rcnn_fpn_4e_mot20-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| R50-FasterRCNN-FPN | - | train | train | Y               |   | 45.9 | 34.8 | 519 | 20444 | 2375 | [config](sort_faster-rcnn_fpn_4e_mot20-public.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |
| R50-FasterRCNN-FPN | - | train | train | N               |   | 76.1 | 59.8 | 5797 | 2791 | 1722 | [config](sort_faster-rcnn_fpn_4e_mot20-private.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth) |