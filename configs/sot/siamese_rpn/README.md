# Siamrpn++: Evolution of Siamese Visual Tracking With Very Deep Networks

## Abstract

<!-- [ABSTRACT] -->

Siamese network based trackers formulate tracking as convolutional feature cross-correlation between a target template and a search region. However, Siamese trackers still have an accuracy gap compared with state-of-the-art algorithms and they cannot take advantage of features from deep networks, such as ResNet-50 or deeper. In this work we prove the core reason comes from the lack of strict translation invariance. By comprehensive theoretical analysis and experimental validations, we break this restriction through a simple yet effective spatial aware sampling strategy and successfully train a ResNet-driven Siamese tracker with significant performance gain. Moreover, we propose a new model architecture to perform layer-wise and depth-wise aggregations, which not only further improves the accuracy but also reduces the model size. We conduct extensive ablation studies to demonstrate the effectiveness of the proposed tracker, which obtains currently the best results on five large tracking benchmarks, including OTB2015, VOT2018, UAV123, LaSOT, and TrackingNet.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142985529-0a9b4e18-5476-40c6-8abf-7d68aab1e5c9.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

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

We provide the best model with its configuration and training log.

|        Method         | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                   Config                    |                                                                                                                                              Download                                                                                                                                              |
| :-------------------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :-----------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       SiamRPN++       |   R-50   |   -   |   20e   |   7.54   |      50.0      |  50.4   |      59.6      |   49.7    |   [config](siamese_rpn_r50_20e_lasot.py)    | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845.log.json) |
| SiamRPN++ <br> (FP16) |   R-50   |   -   |   20e   |    -     |       -        |  50.4   |      59.6      |   49.2    | [config](siamese_rpn_r50_fp16_20e_lasot.py) |                                [model](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_20e_lasot_20220422_181501-ce30fdfd.pth) \| [log](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_20e_lasot_20220422_181501.log.json)                                |

Note:

- `FP16` means Mixed Precision (FP16) is adopted in training.

### UAV123

The checkpoints from 10-th to 20-th epoch will be evaluated during training. You can find the best checkpoint from the log file.

If you want to get better results, you can use the best checkpoint to search the hyperparameters on UAV123 following [here](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/useful_tools_scripts.md#siameserpn-test-time-parameter-search).
Experimentally, the hyperparameters search on UAV123 can bring around 1.0 Success gain.

The results below are achieved without hyperparameters search.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm Precision | Precision |                 Config                  |                                                                                                                                                Download                                                                                                                                                |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :-------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SiamRPN++ |   R-50   |   -   |   20e   |   7.54   |       -        |   60    |      77.3      |   80.3    | [config](siamese_rpn_r50_20e_uav123.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_uav123/siamese_rpn_r50_20e_uav123_20220420_181845-dc2d4831.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_uav123/siamese_rpn_r50_20e_uav123_20220420_181845.log.json) |

### TrackingNet

The results of SiameseRPN++ in TrackingNet are reimplemented by ourselves. The best model on LaSOT is submitted to [the evaluation server on TrackingNet Challenge](http://eval.tracking-net.org/web/challenges/challenge-page/39/submission). We provide the best model with its configuration and training log.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                    Config                    |                                                                                                                                              Download                                                                                                                                              |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SiamRPN++ |   R-50   |   -   |   20e   |   7.54   |       -        |  68.8   |      75.9      |   63.2    | [config](siamese_rpn_r50_20e_trackingnet.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845.log.json) |

### OTB100

The checkpoints from 10-th to 20-th epoch will be evaluated during training. You can find the best checkpoint from the log file.

If you want to get better results, you can use the best checkpoint to search the hyperparameters on OTB100 following [here](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/useful_tools_scripts.md#siameserpn-test-time-parameter-search). Experimentally, the hyperparameters search on OTB100 can bring around 1.0 Success gain.

**Note:** The results reported in the paper are 69.6 Success and 91.4 Precision. We train the SiameseRPN++ in the official [pysot](https://github.com/STVIR/pysot) codebase and can not reproduce the same results. We only get 66.1 Success and 86.7 Precision by following the training and hyperparameters searching instructions of pysot, which are lower than those of the paper by 3.5 Succuess and 4.7 Precision respectively. Without hyperparameters search, we get 65.3 Success and 85.8 Precision. In our codebase, the results below are also achieved without hyperparameters search, close to the results reproduced in pysot in the same setting.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm Precision | Precision |                 Config                  |                                                                                                                                                Download                                                                                                                                                |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :-------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SiamRPN++ |   R-50   |   -   |   20e   |    -     |       -        |  64.9   |      82.4      |   86.3    | [config](siamese_rpn_r50_20e_otb100.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_20e_otb100_20220421_144232-6b8f1730.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_20e_otb100_20220421_144232.log.json) |

### VOT2018

The checkpoints from 10-th to 20-th epoch will be evaluated during training. You can find the best checkpoint from the log file.

If you want to get better results, you can use the best checkpoint to search the hyperparameters on VOT2018 following [here](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/useful_tools_scripts.md#siameserpn-test-time-parameter-search).

**Note:** The result reported in the paper is 0.414 EAO. We train the SiameseRPN++ in the official [pysot](https://github.com/STVIR/pysot) codebase and can not reproduce the same result. We only get 0.364 EAO by following the training and hyperparameters searching instructions of pysot, which is lower than that of the paper by 0.05 EAO. Without hyperparameters search, we get 0.346 EAO. In our codebase, the results below are also achieved without hyperparameters search, close to the results reproduced in pysot in the same setting.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) |  EAO  | Accuracy | Robustness |                  Config                  |                                                                                                                                                  Download                                                                                                                                                  |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :---: | :------: | :--------: | :--------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SiamRPN++ |   R-50   |   -   |   20e   |    -     |       -        | 0.348 |  0.588   |   0.295    | [config](siamese_rpn_r50_20e_vot2018.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_vot2018/siamese_rpn_r50_20e_vot2018_20220420_181845-1111f25e.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_vot2018/siamese_rpn_r50_20e_vot2018_20220420_181845.log.json) |
