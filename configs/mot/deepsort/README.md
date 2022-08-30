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

Currently we do not support training ReID models for DeepSORT.
We directly use the ReID model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). These missed features will be supported in the future.

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                       Config                                       |                                                                                                         Download                                                                                                         |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepSORT | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |      13.8      | 56.9 | 63.7 | 69.5 | 15051 | 40311 | 3312  | [config](deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |

## Get started

### 1. Training

We implement DeepSORT with independent detector and ReID models.
Note that, due to the influence of parameters such as learning rate in default configuration file,
we recommend using 8 GPUs for training in order to reproduce accuracy.

You can train the detector as follows.

```shell script
# Training Faster R-CNN on mot17-half-train dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_train.sh \
    configs/det/faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 2. Testing and evaluation

**2.1 Example on MOTxx-halfval dataset**

```shell script
# Example 1: Test on motXX-half-val set.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/mot/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

If you want to use your own detector checkpoint, you can change the cfg as follows.

```shell script
model = dict(
    detector=dict(
        init_cfg=dict(
            checkpoint=  # noqa: E251
            'path_to_your_checkpoint.pth'  # noqa: E501
        )))
```

Or, you can specify it in commands as follows.

```shell script
./tools/dist_test.sh \
    configs/mot/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 \
    --cfg-options model.detector.init_cfg.checkpoint=path_to_your_checkpoint.pth
```

**2.2 Example on MOTxx-test dataset**

If you want to get the results of the [MOT Challenge](https://motchallenge.net/) test set,
please use the following command to generate result files that can be used for submission.
It will be stored in `./mot_17_test_res`, you can modify the saved path in `test_evaluator` of the config.

```shell script
# Example 2: Test on motxx-test set
# The number after config file represents the number of GPUs used
./tools/dist_test.sh \
    configs/mot/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17train_test-mot17test.py 8
```

If you want to know about more detailed usage of `test.py/dist_test.sh/slurm_test.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/demo_mot_vis.py \
    configs/mot/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4
```

If you want to know about more detailed usage of `demo_mot_vis.py`, please refer to this [document](../../../docs/en/user_guides/3_inference.md).
