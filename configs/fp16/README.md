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

## Results and Models on MOT task

|    Method     |    Detector     |  ReID  | Train Set | Test Set | Public | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
|    :-------:    | :-------------: | :----: | :-------: | :------: | :----: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
|    Tracktor    | R50-FasterRCNN-FPN | R50 | half-train | half-val | N     | -  | 64.7 | 66.6 | 10710 | 45270 | 1152 | [config](tracktor_faster-rcnn_r50_fpn_fp16_4e_mot17-private-half.py) | [detector](https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436-f4ba7d61.pth) &#124; [detector_log](https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436.log.json) &#124; [reid](https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055-4747ee95.pth) &#124; [reid_log](https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055.log.json) |

## Results and Models on SOT task

|    Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Config | Download |
|    :-------:    | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :------: | :--------: |
|    SiameseRPN++    |    R-50    |  -  |   20e    | -        | -              | 49.1 | 57.0 | [config](siamese_rpn_r50_fp16_1x_lasot.py) | [model](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_1x_lasot_20210731_110245-6733c67e.pth) &#124; [log](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_1x_lasot_20210731_110245.log.json) |
