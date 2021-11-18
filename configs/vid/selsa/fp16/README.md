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

## Results and Models on VID task

|    Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 | Config | Download |
|    :-------:    | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    SELSA    |    R-50-DC5     |  pytorch  |   7e    | 2.71        | -            | 78.7 | [config](selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid.py) | [model](https://download.openmmlab.com/mmtracking/fp16/selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid_20210728_193846-dce6eb09.pth) &#124; [log](https://download.openmmlab.com/mmtracking/fp16/selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid_20210728_193846.log.json) |
