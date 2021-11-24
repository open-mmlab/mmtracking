# Siamrpn++: Evolution of Siamese Visual Tracking With Very Deep Networks

## Abstract

Siamese network based trackers formulate tracking as
convolutional feature cross-correlation between a target
template and a search region. However, Siamese track-
ers still have an accuracy gap compared with state-of-the-
art algorithms and they cannot take advantage of features
from deep networks, such as ResNet-50 or deeper. In this
work we prove the core reason comes from the lack of strict
translation invariance. By comprehensive theoretical anal-
ysis and experimental validations, we break this restriction
through a simple yet effective spatial aware sampling strat-
egy and successfully train a ResNet-driven Siamese tracker
with significant performance gain. Moreover, we propose
a new model architecture to perform layer-wise and depth-
wise aggregations, which not only further improves the ac-
curacy but also reduces the model size. We conduct exten-
sive ablation studies to demonstrate the effectiveness of the
proposed tracker, which obtains currently the best results
on five large tracking benchmarks, including OTB2015,
VOT2018, UAV123, LaSOT, and TrackingNet.

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142985529-0a9b4e18-5476-40c6-8abf-7d68aab1e5c9.png"/>
</div>

## Citation

```latex
@inproceedings{li2019siamrpn++,
  title={Siamrpn++: Evolution of siamese visual tracking with very deep networks},
  author={Li, Bo and Wu, Wei and Wang, Qiang and Zhang, Fangyi and Xing, Junliang and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4282--4291},
  year={2019}
}
```

## Results and models

### LaSOT

Note that the checkpoints from 10-th to 20-th epoch will be evaluated during training. You can find the best checkpoint from the log file.

We observe around 1.0 points fluctuations in Success and 1.5 points fluctuations in Norm Precision. We provide the best model with its configuration and training log.

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Config | Download |
| :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :--------: |
| SiamRPN++ |    R-50    |  -  |   20e    | 7.54        | 50.0              | 49.9 | 57.9 | [config](siamese_rpn_r50_1x_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth) &#124; [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20201218_051019.log.json) |
| SiamRPN++ <br> (FP16) |    R-50    |  -  |   20e    | -        | -              | 49.1 | 57.0 | [config](siamese_rpn_r50_fp16_1x_lasot.py) | [model](https://download.openmmlab.com/mmtracking/siamese_rpn_r50_fp16_1x_lasot_20210731_110245-6733c67e.pth) &#124; [log](https://download.openmmlab.com/mmtracking/siamese_rpn_r50_fp16_1x_lasot_20210731_110245.log.json) |

Note:

+ `FP16` means Mixed Precision (FP16) is adopted in training.

### UAV123

The checkpoints from 10-th to 20-th epoch will be evaluated during training. You can find the best checkpoint from the log file.

If you want to get better results, you can use the best checkpoint to search the hyperparameters on UAV123 following [here](https://github.com/open-mmlab/mmtracking/blob/master/docs/useful_tools_scripts.md#siameserpn-test-time-parameter-search).
Experimentally, the hyperparameters search on UAV123 can bring around 1.0 Success gain.

The results below are achieved without hyperparameters search. We observe less than 0.5 points fluctuations both in Success and Precision.

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm Precision | Precision | Config | Download |
| :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :------: | :--------: |
| SiamRPN++ |    R-50    |  -  |   20e    | 7.54     | -             | 60.6 | 76.5 | 80.5 | [config](siamese_rpn_r50_1x_uav123.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_uav123/siamese_rpn_r50_1x_uav123_20210917_104452-36ac4934.pth) &#124; [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_uav123/siamese_rpn_r50_1x_uav123_20210917_104452.log.json) |

### TrackingNet

The results of SiameseRPN++ in TrackingNet are reimplemented by ourselves. The best model on LaSOT is submitted to [the evaluation server on TrackingNet Challenge](http://eval.tracking-net.org/web/challenges/challenge-page/39/submission). We observe less than 0.5 points fluctuations both in Success and Precision. We provide the best model with its configuration and training log.

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |Config | Download |
| :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :------: | :--------: |
| SiamRPN++ |    R-50    |  -  |   20e    |  7.54     | -             | 70.6 | 77.6 | 65.7 | [config](siamese_rpn_r50_1x_trackingnet.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth) &#124; [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20201218_051019.log.json) |

### OTB100

The checkpoints from 10-th to 20-th epoch will be evaluated during training. You can find the best checkpoint from the log file.

If you want to get better results, you can use the best checkpoint to search the hyperparameters on OTB100 following [here](https://github.com/open-mmlab/mmtracking/blob/master/docs/useful_tools_scripts.md#siameserpn-test-time-parameter-search). Experimentally, the hyperparameters search on OTB100 can bring around 1.0 Success gain.

**Note:** We train the SiameseRPN++ in the official [pysot](https://github.com/STVIR/pysot) codebase and can not reproduce the same results reported in the paper. We only get 66.1 Success and 86.7 Precision by following the training and hyperparameters searching instructions of pysot, which are lower than those of the paper by 3.5 Succuess and 4.7 Precision respectively. In our codebase, the Success and Precision are lower 4.8 and 3.7 respectively than those of the paper. Notably, the results below are achieved without hyperparameters search. We observe around 0.5 points fluctuations both in Success and Precision.

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm Precision | Precision | Config | Download |
| :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :------: | :--------: |
| SiamRPN++ |    R-50    |  -  |   20e    |  -   | -              | 64.8 | 83.2 | 87.7 | [config](siamese_rpn_r50_1x_otb100.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_1x_otb100_20210920_001757-12636a0a.pth) &#124; [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_1x_otb100_20210920_001757.log.json) |
