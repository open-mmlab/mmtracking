# Tracking without Bells and Whistles

## Introduction

```latex
@inproceedings{bergmann2019tracking,
  title={Tracking without bells and whistles},
  author={Bergmann, Philipp and Meinhardt, Tim and Leal-Taixe, Laura},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={941--951},
  year={2019}
}
```

## Results and models on MOT17

|    Detector     |  ReID  | Train Set | Test Set | Public | Mem (GB) | Inf time (fps) | MOTA | IDF1 | FP | FN | IDSw. | Config | Download |
| :-------------: | :----: | :-------: | :------: | :----: | :------: | :------------: | :--: | :--: |:--:|:--:| :---: | :----: | :------: |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | Y     |   |   | 57.3 | 63.4 | 1254 | 67091 | 613 |
| R50-FasterRCNN-FPN | R50 | half-train | half-val | N     |   |   | 64.1 | 66.5 | 11088 | 45762 | 1224 |
| R50-FasterRCNN-FPN | R50 | train | train | Y     |   |   | 69.3 | 69.3 | 4010 | 97918 | 1527 |
| R50-FasterRCNN-FPN | R50 | train | train | N     |   |   | 82.1 | 73.4 | 12789 | 44631 | 2988 |
| R50-FasterRCNN-FPN | R50 | train | test  | Y     |   |   | 61.2 | 58.4 | 8612 | 207628 | 2637 |
| R50-FasterRCNN-FPN* | R50 | train | test  | Y     |   |   | 56.3 | 55.1 | 8866 | 235449 | 1987 |
