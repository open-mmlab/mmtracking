# Temporal RoI Align for Video Object Recognition

## Abstract

<!-- [ABSTRACT] -->

Video object detection is challenging in the presence of appearance deterioration in certain video frames. Therefore, it is a natural choice to aggregate temporal information from other frames of the same video into the current frame. However, ROI Align, as one of the most core procedures of video detectors, still remains extracting features from a single-frame feature map for proposals, making the extracted ROI features lack temporal information from videos. In this work, considering the features of the same object instance are highly similar among frames in a video, a novel Temporal ROI Align operator is proposed to extract features from other frames feature maps for current frame proposals by utilizing feature similarity. The proposed Temporal ROI Align operator can extract temporal information from the entire video for proposals. We integrate it into single-frame video detectors and other state-of-the-art video detectors, and conduct quantitative experiments to demonstrate that the proposed Temporal ROI Align operator can consistently and significantly boost the performance. Besides, the proposed Temporal ROI Align can also be applied into video instance segmentation.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142986136-0eebcb9c-0534-4b54-9da9-3f7ea207bd7c.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{gong2021temporal,
  title={Temporal ROI Align for Video Object Recognition},
  author={Gong, Tao and Chen, Kai and Wang, Xinjiang and Chu, Qi and Zhu, Feng and Lin, Dahua and Yu, Nenghai and Feng, Huamin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={2},
  pages={1442--1450},
  year={2021}
}
```

## Results and models on ImageNet VID dataset

We observed that the performance of this method has a fluctuation of about 0.5 mAP. The checkpoint provided below is the best one from two experiments.

Note that the numbers of selsa modules in this method and `SELSA` are 3 and 2 respectively. This is because another selsa modules improve this method by 0.2 points but degrade `SELSA` by 0.5 points. We choose the best settings for the two methods for a fair comparison.

|       Method       | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 |                              Config                              |                                                                                                                                                                                                          Download                                                                                                                                                                                                          |
| :----------------: | :-------: | :-----: | :-----: | :------: | :------------: | :-------: | :--------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Temporal RoI Align | R-50-DC5  | pytorch |   7e    |   4.14   |       -        |   79.8    | [config](selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py)  |   [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714.log.json)   |
| Temporal RoI Align | R-101-DC5 | pytorch |   7e    |   5.83   |       -        |   82.6    | [config](selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621-22cb96b9.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621.log.json) |
| Temporal RoI Align | X-101-DC5 | pytorch |   7e    |   9.74   |       -        |   84.1    | [config](selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036-4471ac42.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036.log.json) |
