## Dataset Preparation

This page provides the instructions for dataset preparation on existing benchmarks, include

- Video Object Detection
  - [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)
- Multiple Object Tracking
  - [MOT Challenge](https://motchallenge.net/)
- Single Object Tracking
  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

### 1. Download Datasets

Please download the datasets from the offical websites. It is recommended to symlink the root of the datasets to `$MMTRACKING/data`. If your folder structure is different from the following, you may need to change the corresponding paths in config files.

Notes:

- The `Lists` under `ILSVRC` contains the txt files from [here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

- For the training and testing of video object detection task, only ILSVRC dataset is needed.

- For the training and testing of multi object tracking task, only MOT17 dataset is needed.

- For the training and testing of single object tracking task, the MSCOCO, ILSVRC and LaSOT datasets are needed.

```
mmtracking
├── mmtrack
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── annotations
│   ├── ILSVRC
│   │   ├── Data
│   │   │   ├── DET
|   │   │   │   ├── train
|   │   │   │   ├── val
|   │   │   │   ├── test
│   │   │   ├── VID
|   │   │   │   ├── train
|   │   │   │   ├── val
|   │   │   │   ├── test
│   │   ├── Annotations (the offical annotation files)
│   │   │   ├── DET
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   │   ├── VID
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   ├── Lists
│   │   ├── annotations (the converted annotation files)
│   ├── lasot
│   │   ├── LaSOTTesting
│   │   ├── annotations
|   ├── MOT17
|   |   ├── train
|   |   ├── test
|   |   ├── annotations
|   |   ├── reid
```

### 2. Convert Annotations

We use [CocoVID](../mmtrack/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase.
In this case, you need to convert the offical annotations to this style. We provide scripts and the usages as follow

```shell
# ImageNet DET
python ./tools/convert_datasets/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# LaSOT
python ./tools/convert_datasets/lasot2coco.py -i ./data/lasot/LaSOTTesting -o ./data/lasot/annotations

# MOT17
python ./tools/convert_datasets/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/convert_datasets/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3
```
