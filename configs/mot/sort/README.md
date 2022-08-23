# Simple online and realtime tracking

## Abstract

<!-- [ABSTRACT] -->

This paper explores a pragmatic approach to multiple object tracking where the main focus is to associate objects efficiently for online and realtime applications. To this end, detection quality is identified as a key factor influencing tracking performance, where changing the detector can improve tracking by up to 18.9%. Despite only using a rudimentary combination of familiar techniques such as the Kalman Filter and Hungarian algorithm for the tracking components, this approach achieves an accuracy comparable to state-of-the-art online trackers. Furthermore, due to the simplicity of our tracking method, the tracker updates at a rate of 260 Hz which is over 20x faster than other state-of-the-art trackers.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/99722489/176848133-d6621813-7b8f-4b25-96cd-2fbcc87983ce.png"/>
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
```

## Results and models on MOT17

We implement SORT with independent detector models. To train a model by yourself, you need to train a detector following [here](../../det/).
The configs in this folder are basically for inference.

| Method |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                     Config                                     |                                                       Download                                                       |
| :----: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :----------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
|  SORT  | R50-FasterRCNN-FPN |  -   | half-train | half-val |   N    |      18.6      | 52.0 | 62.0 | 57.8 | 15150 | 40410 | 5847  | [config](sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |
