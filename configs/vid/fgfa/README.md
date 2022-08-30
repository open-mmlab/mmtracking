# Flow-guided Feature Aggregation for Video Object Detection

## Abstract

<!-- [ABSTRACT] -->

Extending state-of-the-art object detectors from image to video is challenging. The accuracy of detection suffers from degenerated object appearances in videos, e.g., motion blur, video defocus, rare poses, etc. Existing work attempts to exploit temporal information on box level, but such methods are not trained end-to-end. We present flowguided feature aggregation, an accurate and end-to-end learning framework for video object detection. It leverages temporal coherence on feature level instead. It improves the per-frame features by aggregation of nearby features along the motion paths, and thus improves the video recognition accuracy. Our method significantly improves upon strong single-frame baselines in ImageNet VID, especially for more challenging fast moving objects. Our framework is principled, and on par with the best engineered systems winning the ImageNet VID challenges 2016, without additional bells-and-whistles. The proposed method, together with Deep Feature Flow, powered the winning entry of ImageNet VID challenges 2017.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142985575-4560a7c1-0402-428f-9094-ffb00d6b1e38.png"/>
</div>

## Citatioin

<!-- [ALGORITHM] -->

```latex
@inproceedings{zhu2017flow,
  title={Flow-guided feature aggregation for video object detection},
  author={Zhu, Xizhou and Wang, Yujie and Dai, Jifeng and Yuan, Lu and Wei, Yichen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={408--417},
  year={2017}
}
```

## Results and models on ImageNet VID dataset

We observe around 1 mAP fluctuations in performance, and provide the best model.

| Method | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 |                           Config                           |                                                                                                                                                                      Download                                                                                                                                                                      |
| :----: | :-------: | :-----: | :-----: | :------: | :------------: | :-------: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  FGFA  | R-50-DC5  | pytorch |   7e    |   4.10   |      6.9       |   74.7    | [config](fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py)  |   [model](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid_20201228_022657-f42016f3.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid_20201228_022657.log.json)   |
|  FGFA  | R-101-DC5 | pytorch |   7e    |   5.80   |      6.4       |   77.8    | [config](fgfa_faster-rcnn_r101-dc5_8xb1-7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid_20201219_011831-9c9d8183.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid_20201219_011831.log.json) |
|  FGFA  | X-101-DC5 | pytorch |   7e    |   9.74   |       -        |   79.6    | [config](fgfa_faster-rcnn_x101-dc5_8xb1-7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid_20210818_223334-8723c594.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/fgfa/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid/fgfa_faster_rcnn_x101_dc5_1x_imagenetvid_20210818_223334.log.json) |

## Get started

### 1. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. You can use the following command to start the training.

```shell
# The number after config file represents the number of GPUs used. Here we use 8 GPUs
./tools/dist_train.sh \
    configs/vid/fgfa/fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 2. Testing and evaluation

```shell
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/vid/fgfa/fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py 8 \
    --checkpoint ./checkpoints/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid_20201228_022657-f42016f3.pth
```

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/demo_vid.py \
    configs/vid/fgfa/fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py \
    --checkpoint ./checkpoints/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid_20201228_022657-f42016f3.pth \
    --input demo/demo.mp4 \
    --output vid.mp4
```

If you want to know about more detailed usage of `demo_vid.py`, please refer to this [document](../../../docs/en/user_guides/3_inference.md).
