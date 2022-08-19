# Tracking without Bells and Whistles

## Abstract

<!-- [ABSTRACT] -->

The problem of tracking multiple objects in a video sequence poses several challenging tasks. For tracking-by-detection, these include object re-identification, motion prediction and dealing with occlusions. We present a tracker (without bells and whistles) that accomplishes tracking without specifically targeting any of these tasks, in particular, we perform no training or optimization on tracking data. To this end, we exploit the bounding box regression of an object detector to predict the position of an object in the next frame, thereby converting a detector into a Tracktor. We demonstrate the potential of Tracktor and provide a new state-of-the-art on three multi-object tracking benchmarks by extending it with a straightforward re-identification and camera motion compensation. We then perform an analysis on the performance and failure cases of several state-of-the-art tracking methods in comparison to our Tracktor. Surprisingly, none of the dedicated tracking methods are considerably better in dealing with complex tracking scenarios, namely, small and occluded objects or missing detections. However, our approach tackles most of the easy tracking scenarios. Therefore, we motivate our approach as a new tracking paradigm and point out promising future research directions. Overall, Tracktor yields superior tracking performance than any current tracking method and our analysis exposes remaining and unsolved tracking challenges to inspire future research directions.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142983507-fcf71ca3-82c2-4e36-9840-3115476ee23f.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

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
The configs in this folder are basically for inference.

## Results and models on MOT15

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN  | IDSw. |                             Config                              |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :--: | :---: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Tracktor | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 54.3 | 66.6 | 68.3 | 3052 | 3957 |  178  | [config](tracktor_faster-rcnn-resnet50-fpn_8x2bs-4e_mot15halftrain_test-mot15halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040-ae733d0c.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040.log.json) \| [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15_20210803_192157-65b5e2d7.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15_20210803_192157.log.json) |

## Results and models on MOT16

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN   | IDSw. |                             Config                              |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :---: | :---: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Tracktor | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 55.0 | 63.4 | 66.2 | 4179 | 14910 |  444  | [config](tracktor_faster-rcnn-resnet50-fpn_8x2bs-4e_mot16halftrain_test-mot16halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half_20210804_001054-73477869.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half_20210804_001054.log.json) \| [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16_20210803_204826-1b3e3cfd.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16_20210803_204826.log.json) |

## Results and models on MOT17

|        Method        |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                Config                                |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------------------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       Tracktor       | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |      3.1       | 55.8 | 64.1 | 67.0 | 11109 | 45771 | 1227  |   [config](tracktor_faster-rcnn-resnet50-fpn_8x2bs-4e_mot17halftrain_test-mot17halfval.py)    |                                                                                                                                             [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth)                                                                                                                                             |
| Tracktor <br> (FP16) | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 55.5 | 64.7 | 66.7 | 10668 | 45279 | 1185  | [config](tracktor_faster-rcnn-resnet50-fpn_8x2bs-amp-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436-f4ba7d61.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436.log.json) \| [reid](https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055-4747ee95.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055.log.json) |

Note:

- `FP16` means Mixed Precision (FP16) is adopted in training.

## Results and models on MOT20

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |   FN   | IDSw. |                             Config                              |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :----: | :---: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Tracktor | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 52.4 | 70.9 | 64.1 | 5544 | 171729 | 1618  | [config](tracktor_faster-rcnn-resnet50-fpn_8x2bs-8e_mot20halftrain_test-mot20halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244-2c323fd1.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244.log.json) \| [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426.log.json) |
