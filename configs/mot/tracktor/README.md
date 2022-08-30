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

## Results and models on MOT15

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN  | IDSw. |                                       Config                                       |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :--: | :---: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Tracktor | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 54.3 | 66.6 | 68.3 | 3052 | 3957 |  178  | [config](tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot15halftrain_test-mot15halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040-ae733d0c.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040.log.json) \| [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15_20210803_192157-65b5e2d7.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15_20210803_192157.log.json) |

## Results and models on MOT16

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN   | IDSw. |                                       Config                                       |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :---: | :---: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Tracktor | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 55.0 | 63.4 | 66.2 | 4179 | 14910 |  444  | [config](tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot16halftrain_test-mot16halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half_20210804_001054-73477869.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half_20210804_001054.log.json) \| [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16_20210803_204826-1b3e3cfd.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16_20210803_204826.log.json) |

## Results and models on MOT17

|        Method        |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                         Config                                         |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------------------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       Tracktor       | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |      3.1       | 55.8 | 64.1 | 67.0 | 11109 | 45771 | 1227  |   [config](tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py)   |                                                                                                                                             [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth)                                                                                                                                             |
| Tracktor <br> (FP16) | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 55.5 | 64.7 | 66.7 | 10668 | 45279 | 1185  | [config](tracktor_faster-rcnn_r50_fpn_8xb2-amp-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436-f4ba7d61.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436.log.json) \| [reid](https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055-4747ee95.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055.log.json) |

Note:

- `FP16` means Mixed Precision (FP16) is adopted in training.

## Results and models on MOT20

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |   FN   | IDSw. |                                       Config                                       |                                                                                                                                                                                                                                                Download                                                                                                                                                                                                                                                |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :----: | :---: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Tracktor | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |       -        | 52.4 | 70.9 | 64.1 | 5544 | 171729 | 1618  | [config](tracktor_faster-rcnn_r50_fpn_8xb2-8e_mot20halftrain_test-mot20halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244-2c323fd1.pth) \| [detector_log](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244.log.json) \| [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth) \| [reid_log](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426.log.json) |

## Get started

### 1. Training

We implement Tracktor with independent detector and ReID models.
Note that, due to the influence of parameters such as learning rate in default configuration file,
we recommend using 8 GPUs for training in order to reproduce accuracy.

You can train the detector as follows.

```shell script
# Training Faster R-CNN on mot17-half-train dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_train.sh \
    configs/det/faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

And you can train the ReID model as follows.

```shell script
# Training ReID model on mot17-train80 dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_train.sh \
    configs/reid/reid_r50_8xb32-6e_mot17train80_test-mot17val20.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 2. Testing and evaluation

**2.1 Example on MOTxx-halfval dataset**

```shell script
# Example 1: Test on motXX-half-val set.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

If you want to use your own detector and ReID checkpoint, you can change the cfg as follows.

```shell script
model = dict(
    detector=dict(
        init_cfg=dict(
            checkpoint=  # noqa: E251
            'path_to_your_det_checkpoint.pth'  # noqa: E501
        )),
    reid=dict(
        init_cfg=dict(
            checkpoint=  # noqa: E251
            'path_to_your_reid_checkpoint.pth'  # noqa: E501
        )))
```

Or, you can specify them in commands as follows.

```shell script
./tools/dist_test.sh \
    configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 \
    --cfg-options model.detector.init_cfg.checkpoint=path_to_your_det_checkpoint.pth model.reid.init_cfg.checkpoint=path_to_your_reid_checkpoint.pth
```

**2.2 Example on MOTxx-test dataset**

If you want to get the results of the [MOT Challenge](https://motchallenge.net/) test set,
please use the following command to generate result files that can be used for submission.
It will be stored in `./mot_17_test_res`, you can modify the saved path in `test_evaluator` of the config.

```shell script
# Example 2: Test on motxx-test set
# The number after config file represents the number of GPUs used
./tools/dist_test.sh \
    configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot17train_test-mot17test.py 8
```

If you want to know about more detailed usage of `test.py/dist_test.sh/slurm_test.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/demo_mot_vis.py \
    configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4
```

If you want to know about more detailed usage of `demo_mot_vis.py`, please refer to this [document](../../../docs/en/user_guides/3_inference.md).
