# Deep SORT

## Introduction

```latex
@inproceedings{bewley2016simple,
  title={Simple online and realtime tracking},
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  pages={3464--3468},
  year={2016},
  organization={IEEE}
}
@inproceedings{wojke2017simple,
  title={Simple online and realtime tracking with a deep association metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE international conference on image processing (ICIP)},
  pages={3645--3649},
  year={2017},
  organization={IEEE}
}
```

## Results and models on MOT17

|    Detector     |  ReID  | Train Set | Test Set | Public | Mem (GB) | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | -  | half-train | half-val | Y     |   |   |
| R50-FasterRCNN-FPN | -  | half-train | half-val | N     |   |   |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   |   |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   |   |
| R50-FasterRCNN-FPN | - | train | train | Y     |   |   |
| R50-FasterRCNN-FPN | - | train | train | N     |   |   |
| R50-FasterRCNN-FPN | - | train | test  | Y     |   |   |
