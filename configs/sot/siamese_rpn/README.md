# Siamrpn++: Evolution of Siamese Visual Tracking With Very Deep Networks

## Introduction

[ALGORITHM]

```latex
@inproceedings{li2019siamrpn++,
  title={Siamrpn++: Evolution of siamese visual tracking with very deep networks},
  author={Li, Bo and Wu, Wei and Wang, Qiang and Zhang, Fangyi and Xing, Junliang and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4282--4291},
  year={2019}
}
```

## Results and models on LaSOT dataset

We observe around 1.0 points fluctuations in Success and 1.5 points fluctuations in Norm percision. We provide the best model.

Note that all of checkpoints from 11-th to 20-th epoch need to be evaluated in order to achieve the best results.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :--------: |
|    R-50    |  -  |   20e    | 7.54        | 50.0              | 49.9 | 57.9 | [config](siamese_rpn_r50_1x_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth) &#124; [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20201218_051019.log.json) |
