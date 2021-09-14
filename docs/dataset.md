## Dataset Preparation

This page provides the instructions for dataset preparation on existing benchmarks, include

- Video Object Detection
  - [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)
- Multiple Object Tracking
  - [MOT Challenge](https://motchallenge.net/)
- Single Object Tracking
  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
  - [UAV123](https://cemse.kaust.edu.sa/ivul/uav123/)
  - [TrackingNet](https://tracking-net.org/)

### 1. Download Datasets

Please download the datasets from the offical websites. It is recommended to symlink the root of the datasets to `$MMTRACKING/data`. If your folder structure is different from the following, you may need to change the corresponding paths in config files.

Notes:

- The `Lists` under `ILSVRC` contains the txt files from [here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

- For the training and testing of video object detection task, only ILSVRC dataset is needed.

- For the training and testing of multi object tracking task, only one of the MOT Challenge dataset (e.g. MOT17) is needed.

- For the training and testing of single object tracking task, the MSCOCO, ILSVRC, LaSOT, UAV123 and TrackingNet datasets are needed.

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
│   │
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
│   │   ├── Annotations
│   │   │   ├── DET
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   │   ├── VID
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   ├── Lists
│   │
│   ├── lasot
│   │   ├── LaSOTTesting
│   │   │   ├── airplane-1
│   │   │   ├── airplane-13
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
│   │
│   ├── UAV123
│   │   ├── data_seq
│   │   │   ├── UAV123
│   │   │   │   ├── bike1
│   │   │   │   ├── boat1
│   │   ├── anno
│   │   │   ├── UAV123
│   │
│   ├── trackingnet
│   │   ├── TEST
│   │   │   ├── anno
│   │   │   ├── zips
```

### 2. Convert Annotations

We use [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase.
In this case, you need to convert the offical annotations to this style. We provide scripts and the usages are as following:

```shell
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# LaSOT
python ./tools/convert_datasets/lasot/lasot2coco.py -i ./data/lasot/LaSOTTesting -o ./data/lasot/annotations

# MOT17
# The processing of other MOT Challenge dataset is the same as MOT17
python ./tools/convert_datasets/mot/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/convert_datasets/mot/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# UAV123
python ./tools/convert_datasets/uav123/uav2coco.py -i ./data/UAV123/ -o ./data/UAV123/annotations

# TrackingNet
# unzip files in 'TEST/zips/*.zip'
bash ./tools/convert_datasets/trackingnet/unzip_trackingnet_test.sh ./data/trackingnet/TEST
# generate testset annotaions
python ./tools/convert_datasets/trackingnet/trackingnet2coco.py -i ./data/trackingnet/TEST/ -o ./data/trackingnet/TEST/annotations
```

The folder structure will be as following after your run these scripts:

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
│   │
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
│   │
│   ├── lasot
│   │   ├── LaSOTTesting
│   │   │   ├── airplane-1
│   │   │   ├── airplane-13
│   │   ├── annotations
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
|   |   ├── annotations
|   |   ├── reid
│   │   │   ├── imgs
│   │   │   ├── meta
│   │
│   ├── UAV123
│   │   ├── data_seq
│   │   │   ├── UAV123
│   │   │   │   ├── bike1
│   │   │   │   ├── boat1
│   │   ├── anno (the offical annotation files)
│   │   │   ├── UAV123
│   │   ├── annotations (the converted annotation file)
│   │
│   ├── trackingnet
│   │   ├── TEST
│   │   │   ├── anno (the offical annotation files)
│   │   │   ├── zips
│   │   │   ├── annotations (the converted annotation file)
│   │   │   ├── frames (the unzipped folders)
│   │   │   │   ├── 0-6LB4FqxoE_0
│   │   │   │   ├── 07Ysk1C0ZX0_0
```

#### The folder of annotations in ILSVRC

There are 3 json files in `data/ILSVRC/annotations`:

`imagenet_det_30plus1cls.json`: Json file containing the annotations information of the training set in ImageNet DET dataset. The `30` in `30plus1cls` denotes the overlapped 30 categories in ImageNet VID dataset, and the `1cls` means we take the other 170 categories in ImageNet DET dataset as a category, named as `other_categeries`.

`imagenet_vid_train.json`: Json file containing the annotations information of the training set in ImageNet VID dataset.

`imagenet_vid_val.json`: Json file containing the annotations information of the validation set in ImageNet VID dataset.

#### The folder of annotations in lasot

There are only 1 json files in `data/lasot/annotations`:

`lasot_test.json`:  Json file containing the annotations information of the testing set in LaSOT dataset.

#### The folder of annotations and reid in MOT15/MOT16/MOT17/MOT20

We take MOT17 dataset as examples, the other datasets share similar struture.

There are 8 json files in `data/MOT17/annotations`:

`train_cocoformat.json`: Json file containing the annotations information of the training set in MOT17 dataset.

`train_detections.pkl`: Pickle file containing the public detections of the training set in MOT17 dataset.

`test_cocoformat.json`: Json file containing the annotations information of the testing set in MOT17 dataset.

`test_detections.pkl`: Pickle file containing the public detections of the testing set in MOT17 dataset.

`half-train_cocoformat.json`, `half-train_detections.pkl`, `half-val_cocoformat.json`and `half-val_detections.pkl` share similar meaning with `train_cocoformat.json` and `train_detections.pkl`. The `half` means we split each video in the training set into half. The first half videos are denoted as `half-train` set, and the second half videos are denoted as`half-val` set.

The struture of `data/MOT17/reid` is as follows:

```
reid
├── imgs
│   ├── MOT17-02-FRCNN_000002
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
│   ├── MOT17-02-FRCNN_000003
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ...
├── meta
│   ├── train_80.txt
│   ├── val_20.txt
```

The `80` in `train_80.txt` means the proportion of the training dataset to the whole ReID dataset is 80%. While the proportion of the validation dataset is 20%.

For training, we provide a annotation list `train_80.txt`. Each line of the list contains a filename and its corresponding ground-truth labels. The format is as follows:

```
MOT17-05-FRCNN_000110/000018.jpg 0
MOT17-13-FRCNN_000146/000014.jpg 1
MOT17-05-FRCNN_000088/000004.jpg 2
MOT17-02-FRCNN_000009/000081.jpg 3
```

`MOT17-05-FRCNN_000110` denotes the 110-th person in `MOT17-05-FRCNN` video.

For validation, The annotation list `val_20.txt` remains the same as format above.

Images in `reid/imgs` are cropped from raw images in `MOT17/train` by the corresponding `gt.txt`. The value of ground-truth labels should fall in range `[0, num_classes - 1]`.

#### The folder of annotations in UAV123

There are only 1 json files in `data/UAV123/annotations`:

`uav123.json`:  Json file containing the annotations information of the UAV123 dataset.

#### The folder of frames and annotations in TrackingNet

There are 511 video directories of TrackingNet testset in `data/trackingnet/TEST/frames`, and each video directory contains all images of the video.

There are only 1 json files in `data/trackingnet/TEST/annotations`:

`trackingnet_test.json`:  Json file containing the annotations information of the testing set in TrackingNet dataset.
