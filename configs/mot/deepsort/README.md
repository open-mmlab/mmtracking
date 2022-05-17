# Simple online and realtime tracking with a deep association metric

## Abstract

<!-- [ABSTRACT] -->

Simple Online and Realtime Tracking (SORT) is a pragmatic approach to multiple object tracking with a focus on simple, effective algorithms. In this paper, we integrate appearance information to improve the performance of SORT. Due to this extension we are able to track objects through longer periods of occlusions, effectively reducing the number of identity switches. In spirit of the original framework we place much of the computational complexity into an offline pre-training stage where we learn a deep association metric on a largescale person re-identification dataset. During online application, we establish measurement-to-track associations using nearest neighbor queries in visual appearance space. Experimental evaluation shows that our extensions reduce the number of identity switches by 45%, achieving overall competitive performance at high frame rates.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/26813582/145542023-22950508-b35f-41b6-bc78-33d6a82bc3c3.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

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

## Results and models on MOT17

We implement SORT and DeepSORT with independent detector and ReID models. To train a model by yourself, you need to train a detector following [here](../../det/) and also train a ReID model following [here](../../reid).
The configs in this folder are basically for inference.

Currently we do not support training ReID models.
We directly use the ReID model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). These missed features will be supported in the future.

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | MOTA | IDF1 |  FP   |  FN   | IDSw. |                           Config                            |                                                                                                         Download                                                                                                         |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :---: | :---: | :---: | :---------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   SORT   | R50-FasterRCNN-FPN |  -   | half-train | half-val |   Y    |      28.3      | 46.0 | 46.6 |  289  | 82451 | 4581  |   [config](sort_faster-rcnn_fpn_4e_mot17-public-half.py)    |                                                   [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth)                                                   |
|   SORT   | R50-FasterRCNN-FPN |  -   | half-train | half-val |   N    |      18.6      | 62.0 | 57.8 | 15171 | 40437 | 5841  |   [config](sort_faster-rcnn_fpn_4e_mot17-private-half.py)   |                                                   [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth)                                                   |
| DeepSORT | R50-FasterRCNN-FPN | R50  | half-train | half-val |   Y    |      20.4      | 48.1 | 60.8 |  283  | 82445 | 1199  | [config](deepsort_faster-rcnn_fpn_4e_mot17-public-half.py)  | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
| DeepSORT | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |      13.8      | 63.8 | 69.6 | 15060 | 40326 | 3183  | [config](deepsort_faster-rcnn_fpn_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |

Note: When running `demo_mot.py`, we suggest you use the config containing `private`, since `private` means the MOT method doesn't need external detections.
