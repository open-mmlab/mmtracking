# TF-Blender: Temporal Feature Blender for Video Object Detection

## Abstract

<!-- [ABSTRACT] -->

Video objection detection is a challenging task because isolated video frames may encounter appearance deterioration, which introduces great confusion for detection. One of the popular solutions is to exploit the temporal information and enhance per-frame representation through aggregating features from neighboring frames. Despite achieving improvements in detection, existing methods focus on the selection of higher-level video frames for aggregation rather than modeling lower-level temporal relations to increase the feature representation. To address this limitation, we propose a novel solution named TF-Blender, which includes three modules: 1) Temporal relation models the relations between the current frame and its neighboring frames to preserve spatial information. 2). Feature adjustment enriches the representation of every neighboring feature map; 3) Feature blender combines outputs from the first two modules and produces stronger features for the later detection tasks. For its simplicity, TFBlender can be effortlessly plugged into any detection network to improve detection behavior. Extensive evaluations on ImageNet VID and YouTube-VIS benchmarks indicate the performance guarantees of using TF-Blender on recent state-of-the-art methods.

<!-- [IMAGE] -->

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{cui2021tf,
  title={Tf-blender: Temporal feature blender for video object detection},
  author={Cui, Yiming and Yan, Liqi and Cao, Zhiwen and Liu, Dongfang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8138--8147},
  year={2021}
}
```

