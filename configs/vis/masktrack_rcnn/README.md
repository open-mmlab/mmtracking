# Video Instance Segmentation

## Introduction

[ALGORITHM]

```latex
@inproceedings{yang2019video,
  title={Video instance segmentation},
  author={Yang, Linjie and Fan, Yuchen and Xu, Ning},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5188--5197},
  year={2019}
}
```

## Results and models of MaskTrack R-CNN on YouTube-VIS 2019 validation dataset

As mentioned in [Issues #6](https://github.com/youtubevos/MaskTrackRCNN/issues/6#issuecomment-502503505) in MaskTrack R-CNN, the result is kind of unstable for different trials, which ranges from 28 AP to 31 AP when using R-50-FPN as backbone.
The checkpoint provided below is the best one from two experiments.

|    Base detector    |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AP | Config | Download |
| :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    Mask R-CNN    |    R-50-FPN     |  pytorch  |   12e    | 1.61        | -            | 30.7 | [config](masktrack_rcnn_r50_fpn_12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830.log.json) |
|    Mask R-CNN    |    R-101-FPN     |  pytorch  |   12e    |  2.27       | -            | 32.9 | [config](masktrack_rcnn_r101_fpn_12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2019/masktrack_rcnn_r101_fpn_12e_youtubevis2019_20211023_150038-454dc48b.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2019/masktrack_rcnn_r101_fpn_12e_youtubevis2019_20211023_150038.log.json) |
|    Mask R-CNN    |    X-101-FPN     |  pytorch  |   12e    | 3.69        | -            | 35.3 | [config](masktrack_rcnn_x101_fpn_12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2019/masktrack_rcnn_x101_fpn_12e_youtubevis2019_20211023_153205-fff7a102.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2019/masktrack_rcnn_x101_fpn_12e_youtubevis2019_20211023_153205.log.json) |

## Results and models of MaskTrack R-CNN on YouTube-VIS 2021 validation dataset

The checkpoint provided below is the best one from two experiments.

|    Base detector    |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AP | Config | Download |
| :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    Mask R-CNN    |    R-50-FPN     |  pytorch  |   12e    | 1.61        | -            | 29.7 | [config](masktrack_rcnn_r50_fpn_12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2021/masktrack_rcnn_r50_fpn_12e_youtubevis2021_20211026_044948-10da90d9.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2021/masktrack_rcnn_r50_fpn_12e_youtubevis2021_20211026_044948.log.json) |
|    Mask R-CNN    |    R-101-FPN     |  pytorch  |   12e    | 2.27         | -            | 31.6 | [config](masktrack_rcnn_r101_fpn_12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2021/masktrack_rcnn_r101_fpn_12e_youtubevis2021_20211026_045509-3c49e4f3.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2021/masktrack_rcnn_r101_fpn_12e_youtubevis2021_20211026_045509.log.json) |
|    Mask R-CNN    |    X-101-FPN     |  pytorch  |   12e    | 3.69         | -            | 33.7 | [config](masktrack_rcnn_x101_fpn_12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2021/masktrack_rcnn_x101_fpn_12e_youtubevis2021_20211026_095943-90831df4.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2021/masktrack_rcnn_x101_fpn_12e_youtubevis2021_20211026_095943.log.json) |
