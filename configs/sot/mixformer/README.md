# MixFormer: End-to-End Tracking with Iterative Mixed Attention

## Abstract

<!-- [ABSTRACT] -->

Tracking often uses a multi-stage pipeline of feature extraction, target information integration, and bounding box estimation. To simplify this pipeline and unify the process of feature extraction and target information integration, we present a compact tracking framework, termed as MixFormer, built upon transformers. Our core design is to utilize the flexibility of attention operations, and propose a Mixed Attention Module (MAM) for simultaneous feature extraction and target information integration. This synchronous modeling scheme allows to extract target-specific discriminative features and perform extensive communication between target and search area. Based on MAM, we build our MixFormer tracking framework simply by stacking multiple MAMs with progressive patch embedding and placing a localization head on top. In addition, to handle multiple target templates during online tracking, we devise an asymmetric attention scheme in MAM to reduce computational cost, and propose an effective score prediction module to select high-quality templates. Our MixFormer sets a new state-of-the-art performance on five tracking benchmarks, including LaSOT, TrackingNet, VOT2020, GOT-10k, and UAV123. In particular, our MixFormer-L achieves NP score of 79.9% on LaSOT, 88.9% on TrackingNet and EAO of 0.555 on VOT2020. We also perform in-depth ablation studies to demonstrate the effectiveness of simultaneous feature extraction and information integration. Code and trained models are publicly available at [here](https://github.com/MCG-NJU/MixFormer).

<!-- [IMAGE] -->

![MixFormer architecture](https://user-images.githubusercontent.com/77977134/182669431-68effbcf-6e8c-4c69-8b3e-796e1dfd0f0a.jpg)

## Citation

<!-- [ALGORITHM] -->

```latex
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

|  Method   | Style | Inf time(fps) | Success | Norm precision | Precision |                 Config                  |                                                        Download                                                        |
| :-------: | :---: | :-----------: | :-----: | :------------: | :-------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| MixFormer |   -   |       -       |  69.0   |      79.6      |   75.2    | [config](./mixformer_cvt_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_lasot/mixformer_cvt_500e_lasot.pth) |

### TrackingNet

|  Method   | Style | Inf time(fps) | Success | Norm precision | Precision |                    Config                     |                                                        Download                                                        |
| :-------: | :---: | :-----------: | :-----: | :------------: | :-------: | :-------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| MixFormer |   -   |       -       |  81.4   |      86.8      |   80.3    | [config](./mixformer_cvt_500e_trackingnet.py) | [model](https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_lasot/mixformer_cvt_500e_lasot.pth) |

### GOT10k

|  Method   | Style | Inf time(fps) | Average Overlap | Success Rate 0.5 | Success Rate 0.75 |                  Config                  |                                                         Download                                                         |
| :-------: | :---: | :-----------: | :-------------: | :--------------: | :---------------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| MixFormer |   -   |       -       |      70.1       |       80.1       |       65.6        | [config](./mixformer_cvt_500e_got10k.py) | [model](https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_got10k/mixformer_cvt_500e_got10k.pth) |
