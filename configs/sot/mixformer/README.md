# MixFormer: End-to-End Tracking with Iterative Mixed Attention

## Abstract

<!-- [ABSTRACT] -->
Tracking often uses a multi-stage pipeline of feature extraction, target information integration, and bounding box estimation. To simplify this pipeline and unify the process of feature extraction and target information integration, we present a compact tracking framework, termed as MixFormer, built upon transformers. Our core design is to utilize the flexibility of attention operations, and propose a Mixed Attention Module (MAM) for simultaneous feature extraction and target information integration. This synchronous modeling scheme allows to extract target-specific discriminative features and perform extensive communication between target and search area. Based on MAM, we build our MixFormer tracking framework simply by stacking multiple MAMs with progressive patch embedding and placing a localization head on top. In addition, to handle multiple target templates during online tracking, we devise an asymmetric attention scheme in MAM to reduce computational cost, and propose an effective score prediction module to select high-quality templates. Our MixFormer sets a new state-of-the-art performance on five tracking benchmarks, including LaSOT, TrackingNet, VOT2020, GOT-10k, and UAV123. In particular, our MixFormer-L achieves NP score of 79.9% on LaSOT, 88.9% on TrackingNet and EAO of 0.555 on VOT2020. We also perform in-depth ablation studies to demonstrate the effectiveness of simultaneous feature extraction and information integration. Code and trained models are publicly available at [here](https://github.com/MCG-NJU/MixFormer).

## Citation

<!-- [ALGORITHM] -->

``` latex
@inproceedings{cui2022mixformer,
  title={MixFormer: End-to-End Tracking with Iterative Mixed Attention},
  author={Cui, Yutao and Jiang, Cheng and Wang, Limin and Wu, Gangshan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13608--13618},
  year={2022}
}
```

## Results and models

We now provide the evaluation results using model weights released in [official repository](https://github.com/MCG-NJU/MixFormer).

### Lasot

| Method | Style | Inf time(fps) |  Success | Norm precision | Precision | Config |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|MixFormer | - | - | 69.2 | 78.8 | 74.3 | [config](./mixformer_lasot.py) |

### TrackingNet

| Method | Style | Inf time(fps) |  Success | Norm precision | Precision | Config |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|MixFormer | - | - | 81.1 | 86.6 | 80.0 | [config](./mixformer_trackingnet.py) |

### GOT10k
| Method | Style | Inf time(fps) | Average Overlap | Success Rate 0.5 | Success Rate 0.75 | Config |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|MixFormer | - | - | 72.0 | 82.0 | 68.1 | [config](./mixformer_got10k.py) |

