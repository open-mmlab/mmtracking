# Quasi-Dense Similarity Learning for Multiple Object Tracking

## Abstract

<!-- [ABSTRACT] -->

Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions on the images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of region proposals on a pair of images for contrastive learning. We can directly combine this similarity learning with existing detection methods to build Quasi-Dense Tracking (QDTrack) without turning to displacementregression or motion priors. We also find that the resulting distinctive feature space admits a simple nearest neighbor search at the inference time. Despite its simplicity, QD-Track outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external training data. Compared to methods with similar detectors, it boosts almost 10 points of MOTA and significantly decreases the number of ID switches on BDD100K and Waymo datasets.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/48645550/158332287-79fb379b-d817-4aa8-8530-5f9d172b3ca7.png"/>
  <img src="https://user-images.githubusercontent.com/48645550/158332524-8ccaab0e-d379-4c6b-83e5-d75398af02bf.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{pang2021quasi,
  title={Quasi-dense similarity learning for multiple object tracking},
  author={Pang, Jiangmiao and Qiu, Linlu and Li, Xia and Chen, Haofeng and Li, Qi and Darrell, Trevor and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={164--173},
  year={2021}
}
```

## Results and models on MOT17

| Method  |   Detector   |        Train Set        | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN   | IDSw. |                                  Config                                   |                                                                                                                                                   Download                                                                                                                                                   |
| :-----: | :----------: | :---------------------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :---: | :---: | :-----------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QDTrack | Faster R-CNN |       half-train        | half-val |   N    |       -        | 57.1 | 68.2 | 68.5 | 8373 | 42939 | 1071  |      [config](qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py)       |            [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635.log.json)            |
| QDTrack | Faster R-CNN | CrowdHuman + half-train | half-val |   N    |       -        | 59.1 | 71.7 | 71.6 | 6072 | 38733 |  867  | [config](qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17-private-half.py) | [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453-68899b0a.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453.log.json) |
