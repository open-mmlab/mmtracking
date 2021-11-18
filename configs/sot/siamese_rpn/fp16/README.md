# Mixed Precision Training

## Introduction

<!-- [OTHERS] -->

```latex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
}
```

## Results and Models on SOT task

|    Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Config | Download |
|    :-------:    | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :--------: |
|    SiameseRPN++    |    R-50    |  -  |   20e    | -        | -              | 49.1 | 57.0 | [config](siamese_rpn_r50_fp16_1x_lasot.py) | [model](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_1x_lasot_20210731_110245-6733c67e.pth) &#124; [log](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_1x_lasot_20210731_110245.log.json) |
