# Sequence Level Semantics Aggregation for Video Object Detection

## Abstract

<!-- [ABSTRACT] -->

Video objection detection (VID) has been a rising research direction in recent years. A central issue of VID is the appearance degradation of video frames caused by fast motion. This problem is essentially ill-posed for a single frame. Therefore, aggregating features from other frames becomes a natural choice. Existing methods rely heavily on optical flow or recurrent neural networks for feature aggregation. However, these methods emphasize more on the temporally nearby frames. In this work, we argue that aggregating features in the full-sequence level will lead to more discriminative and robust features for video object detection. To achieve this goal, we devise a novel Sequence Level Semantics Aggregation (SELSA) module. We further demonstrate the close relationship between the proposed method and the classic spectral clustering method, providing a novel view for understanding the VID problem. We test the proposed method on the ImageNet VID and the EPIC KITCHENS dataset and achieve new state-of-theart results. Our method does not need complicated postprocessing methods such as Seq-NMS or Tubelet rescoring, which keeps the pipeline simple and clean.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142985636-ad7b2d17-3d29-4b08-90e6-22f6a29dafaf.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{wu2019sequence,
  title={Sequence level semantics aggregation for video object detection},
  author={Wu, Haiping and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9217--9225},
  year={2019}
}
```

## Results and models on ImageNet VID dataset

We observe around 1 mAP fluctuations in performance, and provide the best model.

|      Method       | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 |                           Config                           |                                                                                                                                                                         Download                                                                                                                                                                         |
| :---------------: | :-------: | :-----: | :-----: | :------: | :------------: | :-------: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       SELSA       | R-50-DC5  | pytorch |   7e    |   3.49   |      7.5       |   78.4    |   [config](selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py)    |   [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835.log.json)   |
|       SELSA       | R-101-DC5 | pytorch |   7e    |   5.18   |      7.2       |   81.5    |   [config](selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py)   | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724.log.json) |
|       SELSA       | X-101-DC5 | pytorch |   7e    |   9.15   |       -        |   83.1    |   [config](selsa_faster_rcnn_x101_dc5_1x_imagenetvid.py)   | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_x101_dc5_1x_imagenetvid/selsa_faster_rcnn_x101_dc5_1x_imagenetvid_20210825_205641-10252965.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_x101_dc5_1x_imagenetvid/selsa_faster_rcnn_x101_dc5_1x_imagenetvid_20210825_205641.log.json) |
| SELSA <br> (FP16) | R-50-DC5  | pytorch |   7e    |   2.71   |       -        |   78.7    | [config](selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid.py) |                                            [model](https://download.openmmlab.com/mmtracking/fp16/selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid_20210728_193846-dce6eb09.pth) \| [log](https://download.openmmlab.com/mmtracking/fp16/selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid_20210728_193846.log.json)                                            |

Note:

- `FP16` means Mixed Precision (FP16) is adopted in training.
