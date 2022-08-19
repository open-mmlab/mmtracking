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

We implement DeepSORT with independent detector and ReID models. To train a model by yourself, you need to train a detector following [here](../../det/) and also train a ReID model following [here](../../reid).
The configs in this folder are basically for inference.

Currently we do not support training ReID models.
We directly use the ReID model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). These missed features will be supported in the future.

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                           Config                            |                                                                                                         Download                                                                                                         |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :---------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepSORT | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |      13.8      | 56.9 | 63.7 | 69.5 | 15051 | 40311 | 3312  | [config](deepsort_faster-rcnn-resnet50-fpn_8x2bs-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |
