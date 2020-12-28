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
| R50-FasterRCNN-FPN | -  | half-train | half-val | Y     |   |   |   46.0 | 46.6 | 289 | 82451 | 4581 |
| R50-FasterRCNN-FPN | -  | half-train | half-val | N     |   |   |   62.0 | 57.8 | 15171 | 40437 | 5841 |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   |   |  48.1 | 60.8 | 283   | 82445 | 1199 |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   |   |  63.8 | 69.6 | 15060 | 40326 | 3183 |
| R50-FasterRCNN-FPN | - | train | train | Y     |   |   | 50.9 | 44.5 | 1108 | 153950 | 10522 |
| R50-FasterRCNN-FPN | - | train | train | N     |   |   | 80.8 | 61.3 | 21537 | 29280 | 13947 |
