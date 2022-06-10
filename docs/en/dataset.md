## Dataset Preparation

This page provides the instructions for dataset preparation on existing benchmarks, include

- Video Object Detection
  - [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)
- Multiple Object Tracking
  - [MOT Challenge](https://motchallenge.net/)
  - [CrowdHuman](https://www.crowdhuman.org/)
  - [LVIS](https://www.lvisdataset.org/)
  - [TAO](https://taodataset.org/)
  - [DanceTrack](https://dancetrack.github.io)
- Single Object Tracking
  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
  - [UAV123](https://cemse.kaust.edu.sa/ivul/uav123/)
  - [TrackingNet](https://tracking-net.org/)
  - [OTB100](http://www.visual-tracking.net/)
  - [GOT10k](http://got-10k.aitestunion.com/)
  - [VOT2018](https://www.votchallenge.net/vot2018/)
- Video Instance Segmentation
  - [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

### 1. Download Datasets

Please download the datasets from the official websites. It is recommended to symlink the root of the datasets to `$MMTRACKING/data`.

#### 1.1 Video Object Detection

- For the training and testing of video object detection task, only ILSVRC dataset is needed.

- The `Lists` under `ILSVRC` contains the txt files from [here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

#### 1.2 Multiple Object Tracking

- For the training and testing of multi object tracking task, one of the MOT Challenge datasets (e.g. MOT17, TAO and DanceTrack) are needed, CrowdHuman and LVIS can be served as comlementary dataset.

- The `annotations` under `tao` contains the official annotations from [here](https://github.com/TAO-Dataset/annotations).

- The `annotations` under `lvis` contains the official annotations of lvis-v0.5 which can be downloaded according to [here](https://github.com/lvis-dataset/lvis-api/issues/23#issuecomment-894963957). The synset mapping file `coco_to_lvis_synset.json` used in `./tools/convert_datasets/tao/merge_coco_with_lvis.py` script can be found [here](https://github.com/TAO-Dataset/tao/tree/master/data).

#### 1.3 Single Object Tracking

- For the training and testing of single object tracking task, the MSCOCO, ILSVRC, LaSOT, UAV123, TrackingNet, OTB100, GOT10k and VOT2018 datasets are needed.

- For OTB100 dataset, you don't need to download the dataset from the official website manually, since we provide a script to download it.

```shell
# download OTB100 dataset by web crawling
python ./tools/convert_datasets/otb100/download_otb100.py -o ./data/otb100/zips -p 8
```

- For VOT2018, we use the official downloading script.

```shell
# download VOT2018 dataset by web crawling
python ./tools/convert_datasets/vot/download_vot.py --dataset vot2018 --save_path ./data/vot2018/data
```

#### 1.4 Video Instance Segmentation

- For the training and testing of video instance segmetatioon task, only one of YouTube-VIS datasets (e.g. YouTube-VIS 2019) is needed.

#### 1.5 Data Structure

If your folder structure is different from the following, you may need to change the corresponding paths in config files.

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
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
│   │
|   ├── DanceTrack
|   |   ├── train
|   |   ├── val
|   |   ├── test
|   |
│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip
│   │
│   ├── lvis
│   │   ├── train (the same as coco/train2017)
│   │   ├── val (the same as coco/val2017)
│   │   ├── test (the same as coco/test2017)
│   │   ├── annotations
│   │   │   ├── coco_to_lvis_synset.json
│   │   │   ├── lvis_v0.5_train.json
│   │   │   ├── lvis_v0.5_val.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_image_info_test_challenge.json
│   │   │   ├── lvis_v1_image_info_test_dev.json
│   │
│   ├── tao
│   │   ├── annotations
│   │   │   ├── test_without_annotations.json
│   │   │   ├── train.json
│   │   │   ├── validation.json
│   │   │   ├── ......
│   │   ├── test
│   │   │   ├── ArgoVerse
│   │   │   ├── AVA
│   │   │   ├── BDD
│   │   │   ├── Charades
│   │   │   ├── HACS
│   │   │   ├── LaSOT
│   │   │   ├── YFCC100M
│   │   ├── train
│   │   ├── val
│   │
│   ├── lasot
│   │   ├── LaSOTBenchmark
│   │   │   ├── airplane
|   │   │   │   ├── airplane-1
|   │   │   │   ├── airplane-2
|   │   │   │   ├── ......
│   │   │   ├── ......
│   │
│   ├── UAV123
│   │   ├── data_seq
│   │   │   ├── UAV123
│   │   │   │   ├── bike1
│   │   │   │   ├── boat1
│   │   │   │   ├── ......
│   │   ├── anno
│   │   │   ├── UAV123
│   │
│   ├── trackingnet
│   │   ├── TEST.zip
│   │   ├── TRAIN_0.zip
│   │   ├── ......
│   │   ├── TRAIN_11.zip
│   │
│   ├── otb100
│   │   │── zips
│   │   │   │── Basketball.zip
│   │   │   │── Biker.zip
│   │   │   │──
│   │
│   ├── got10k
│   │   │── full_data
│   │   │   │── train_data
│   │   │   │   ├── GOT-10k_Train_split_01.zip
│   │   │   │   ├── ......
│   │   │   │   ├── GOT-10k_Train_split_19.zip
│   │   │   │   ├── list.txt
│   │   │   │── test_data.zip
│   │   │   │── val_data.zip
│   │
|   ├── vot2018
|   |   ├── data
|   |   |   ├── ants1
|   │   │   │   ├──color
│   │
│   ├── youtube_vis_2019
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── train.json (the official annotation files)
│   │   │── valid.json (the official annotation files)
│   │   │── test.json (the official annotation files)
│   │
│   ├── youtube_vis_2021
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
```

### 2. Convert Annotations

We use [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase.
In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```shell
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# MOT17
# The processing of other MOT Challenge dataset is the same as MOT17
python ./tools/convert_datasets/mot/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/convert_datasets/mot/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# DanceTrack
python ./tools/convert_datasets/dancetrack/dancetrack2coco.py -i ./data/DanceTrack ./data/DanceTrack/annotations

# CrowdHuman
python ./tools/convert_datasets/mot/crowdhuman2coco.py -i ./data/crowdhuman -o ./data/crowdhuman/annotations

# LVIS
# Merge annotations from LVIS and COCO for training QDTrack
python ./tools/convert_datasets/tao/merge_coco_with_lvis.py --lvis ./data/lvis/annotations/lvis_v0.5_train.json --coco ./data/coco/annotations/instances_train2017.json --mapping ./data/lvis/annotations/coco_to_lvis_synset.json --output-json ./data/lvis/annotations/lvisv0.5+coco_train.json

# TAO
# Generate filtered json file for QDTrack
python ./tools/convert_datasets/tao/tao2coco.py -i ./data/tao/annotations --filter-classes

# LaSOT
python ./tools/convert_datasets/lasot/gen_lasot_infos.py -i ./data/lasot/LaSOTBenchmark -o ./data/lasot/annotations

# UAV123
# download annotations
# due to the annotations of all videos in UAV123 are inconsistent, we just download the information file generated in advance.
wget https://download.openmmlab.com/mmtracking/data/uav123_infos.txt -P data/uav123/annotations

# TrackingNet
# unzip files in 'data/trackingnet/*.zip'
bash ./tools/convert_datasets/trackingnet/unzip_trackingnet.sh ./data/trackingnet
# generate annotations
python ./tools/convert_datasets/trackingnet/gen_trackingnet_infos.py -i ./data/trackingnet -o ./data/trackingnet/annotations

# OTB100
# unzip files in 'data/otb100/zips/*.zip'
bash ./tools/convert_datasets/otb100/unzip_otb100.sh ./data/otb100
# download annotations
# due to the annotations of all videos in OTB100 are inconsistent, we just need to download the information file generated in advance.
wget https://download.openmmlab.com/mmtracking/data/otb100_infos.txt -P data/otb100/annotations

# GOT10k
# unzip 'data/got10k/full_data/test_data.zip', 'data/got10k/full_data/val_data.zip' and files in 'data/got10k/full_data/train_data/*.zip'
bash ./tools/convert_datasets/got10k/unzip_got10k.sh ./data/got10k
# generate annotations
python ./tools/convert_datasets/got10k/gen_got10k_infos.py -i ./data/got10k -o ./data/got10k/annotations

# VOT2018
python ./tools/convert_datasets/vot/gen_vot_infos.py -i ./data/vot2018 -o ./data/vot2018/annotations --dataset_type vot2018

# YouTube-VIS 2019
python ./tools/convert_datasets/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2019 -o ./data/youtube_vis_2019/annotations --version 2019

# YouTube-VIS 2021
python ./tools/convert_datasets/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2021 -o ./data/youtube_vis_2021/annotations --version 2021
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
│   │   ├── Annotations (the official annotation files)
│   │   │   ├── DET
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   │   ├── VID
|   │   │   │   ├── train
|   │   │   │   ├── val
│   │   ├── Lists
│   │   ├── annotations (the converted annotation files)
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
|   |   ├── annotations
|   |   ├── reid
│   │   │   ├── imgs
│   │   │   ├── meta
│   │
│   ├── DanceTrack
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── annotations
│   │
│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip
│   │   ├── annotations
│   │   │   ├── crowdhuman_train.json
│   │   │   ├── crowdhuman_val.json
│   │
│   ├── lvis
│   │   ├── train (the same as coco/train2017)
│   │   ├── val (the same as coco/val2017)
│   │   ├── test (the same as coco/test2017)
│   │   ├── annotations
│   │   │   ├── coco_to_lvis_synset.json
│   │   │   ├── lvisv0.5+coco_train.json
│   │   │   ├── lvis_v0.5_train.json
│   │   │   ├── lvis_v0.5_val.json
│   │   │   ├── lvis_v1_train.json
│   │   │   ├── lvis_v1_val.json
│   │   │   ├── lvis_v1_image_info_test_challenge.json
│   │   │   ├── lvis_v1_image_info_test_dev.json
│   │
│   ├── tao
│   │   ├── annotations
│   │   │   ├── test_482_classes.json
│   │   │   ├── test_without_annotations.json
│   │   │   ├── train.json
│   │   │   ├── train_482_classes.json
│   │   │   ├── validation.json
│   │   │   ├── validation_482_classes.json
│   │   │   ├── ......
│   │   ├── test
│   │   │   ├── ArgoVerse
│   │   │   ├── AVA
│   │   │   ├── BDD
│   │   │   ├── Charades
│   │   │   ├── HACS
│   │   │   ├── LaSOT
│   │   │   ├── YFCC100M
│   │   ├── train
│   │   ├── val
│   │
│   ├── lasot
│   │   ├── LaSOTBenchmark
│   │   │   ├── airplane
|   │   │   │   ├── airplane-1
|   │   │   │   ├── airplane-2
|   │   │   │   ├── ......
│   │   │   ├── ......
│   │   ├── annotations
│   │
│   ├── UAV123
│   │   ├── data_seq
│   │   │   ├── UAV123
│   │   │   │   ├── bike1
│   │   │   │   ├── boat1
│   │   │   │   ├── ......
│   │   ├── anno (the official annotation files)
│   │   │   ├── UAV123
│   │   ├── annotations (the converted annotation file)
│   │
│   ├── trackingnet
│   │   ├── TEST
│   │   │   ├── anno (the official annotation files)
│   │   │   ├── zips
│   │   │   ├── frames (the unzipped folders)
│   │   │   │   ├── 0-6LB4FqxoE_0
│   │   │   │   ├── 07Ysk1C0ZX0_0
│   │   │   │   ├── ......
│   │   ├── TRAIN_0
│   │   │   ├── anno (the official annotation files)
│   │   │   ├── zips
│   │   │   ├── frames (the unzipped folders)
│   │   │   │   ├── -3TIfnTSM6c_2
│   │   │   │   ├── a1qoB1eERn0_0
│   │   │   │   ├── ......
│   │   ├── ......
│   │   ├── TRAIN_11
│   │   ├── annotations (the converted annotation file)
│   │
│   ├── otb100
│   │   ├── zips
│   │   │   ├── Basketball.zip
│   │   │   ├── Biker.zip
│   │   │   │── ......
│   │   ├── annotations
│   │   ├── data
│   │   │   ├── Basketball
│   │   │   │   ├── img
│   │   │   ├── ......
│   │
│   ├── got10k
│   │   │── full_data
│   │   │   │── train_data
│   │   │   │   ├── GOT-10k_Train_split_01.zip
│   │   │   │   ├── ......
│   │   │   │   ├── GOT-10k_Train_split_19.zip
│   │   │   │   ├── list.txt
│   │   │   │── test_data.zip
│   │   │   │── val_data.zip
│   │   │── train
│   │   │   ├── GOT-10k_Train_000001
│   │   │   │   ├── ......
│   │   │   ├── GOT-10k_Train_009335
│   │   │   ├── list.txt
│   │   │── test
│   │   │   ├── GOT-10k_Test_000001
│   │   │   │   ├── ......
│   │   │   ├── GOT-10k_Test_000180
│   │   │   ├── list.txt
│   │   │── val
│   │   │   ├── GOT-10k_Val_000001
│   │   │   │   ├── ......
│   │   │   ├── GOT-10k_Val_000180
│   │   │   ├── list.txt
│   │   │── annotations
│   │
|   ├── vot2018
|   |   ├── data
|   |   |   ├── ants1
|   │   │   │   ├──color
|   |   ├── annotations
│   │   │   ├── ......
│   │
│   ├── youtube_vis_2019
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── ......
│   │   │── train.json (the official annotation files)
│   │   │── valid.json (the official annotation files)
│   │   │── test.json (the official annotation files)
│   │   │── annotations (the converted annotation file)
│   │
│   ├── youtube_vis_2021
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── annotations (the converted annotation file)
```

#### The folder of annotations in ILSVRC

There are 3 JSON files in `data/ILSVRC/annotations`:

`imagenet_det_30plus1cls.json`: JSON file containing the annotations information of the training set in ImageNet DET dataset. The `30` in `30plus1cls` denotes the overlapped 30 categories in ImageNet VID dataset, and the `1cls` means we take the other 170 categories in ImageNet DET dataset as a category, named as `other_categeries`.

`imagenet_vid_train.json`: JSON file containing the annotations information of the training set in ImageNet VID dataset.

`imagenet_vid_val.json`: JSON file containing the annotations information of the validation set in ImageNet VID dataset.

#### The folder of annotations and reid in MOT15/MOT16/MOT17/MOT20

We take MOT17 dataset as examples, the other datasets share similar structure.

There are 8 JSON files in `data/MOT17/annotations`:

`train_cocoformat.json`: JSON file containing the annotations information of the training set in MOT17 dataset.

`train_detections.pkl`: Pickle file containing the public detections of the training set in MOT17 dataset.

`test_cocoformat.json`: JSON file containing the annotations information of the testing set in MOT17 dataset.

`test_detections.pkl`: Pickle file containing the public detections of the testing set in MOT17 dataset.

`half-train_cocoformat.json`, `half-train_detections.pkl`, `half-val_cocoformat.json`and `half-val_detections.pkl` share similar meaning with `train_cocoformat.json` and `train_detections.pkl`. The `half` means we split each video in the training set into half. The first half videos are denoted as `half-train` set, and the second half videos are denoted as`half-val` set.

The structure of `data/MOT17/reid` is as follows:

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

#### The folder of annotations in crowdhuman

There are 2 JSON files in `data/crowdhuman/annotations`:

`crowdhuman_train.json`:  JSON file containing the annotations information of the training set in CrowdHuman dataset.
`crowdhuman_val.json`:  JSON file containing the annotations information of the validation set in CrowdHuman dataset.

#### The folder of annotations in lvis

There are 8 JSON files in `data/lvis/annotations`

`coco_to_lvis_synset.json`: JSON file containing the mapping relationship between COCO and LVIS categories.

`lvisv0.5+coco_train.json`: JSON file containing the merged annotations.

`lvis_v0.5_train.json`: JSON file containing the annotations information of the training set in lvisv0.5.

`lvis_v0.5_val.json`: JSON file containing the annotations information of the validation set in lvisv0.5.

`lvis_v1_train.json`: JSON file containing the annotations information of the training set in lvisv1.

`lvis_v1_val.json`: JSON file containing the annotations information of the validation set in lvisv1.

`lvis_v1_image_info_test_challenge.json`: JSON file containing the annotations information of the testing set in lvisv1 available for year-round evaluation.

`lvis_v1_image_info_test_dev.json`: JSON file containing the annotations information of the testing set in lvisv1 available only once a year for LVIS Challenge.

#### The folder of annotations in tao

There are 9 JSON files in `data/tao/annotations`:

`test_categories.json`: JSON file containing a list of categories which will be evaluated on the TAO test set.

`test_without_annotations.json`:  JSON for test videos. The 'images' and 'videos' fields contain the images and videos that will be evaluated on the test set.

`test_482_classes.json`: JSON file containing the converted results for test set.

`train.json`: JSON file containing annotations for LVIS categories in TAO train.

`train_482_classes.json`: JSON file containing the converted results for train set.

`train_with_freeform.json`: JSON file containing annotations for all categories in TAO train.

`validation.json`: JSON file containing annotations for LVIS categories in TAO train.

`validation_482_classes.json`: JSON file containing the converted results for validation set.

`validation_with_freeform.json`: JSON file containing annotations for all categories in TAO validation.

#### The folder of annotations in lasot

There are 2 JSON files in `data/lasot/annotations`:

`lasot_train.json`:  JSON file containing the annotations information of the training set in LaSOT dataset.
`lasot_test.json`:  JSON file containing the annotations information of the testing set in LaSOT dataset.

There are 2 TEXT files in `data/lasot/annotations`:

`lasot_train_infos.txt`:  TEXT file containing the annotations information of the training set in LaSOT dataset.
`lasot_test_infos.txt`:  TEXT file containing the annotations information of the testing set in LaSOT dataset.

#### The folder of annotations in UAV123

There are only 1 JSON files in `data/UAV123/annotations`:

`uav123.json`:  JSON file containing the annotations information of the UAV123 dataset.

There are only 1 TEXT files in `data/UAV123/annotations`:

`uav123_infos.txt`:  TEXT file containing the information of the UAV123 dataset.

#### The folder of frames and annotations in TrackingNet

There are 511 video directories of TrackingNet testset in `data/trackingnet/TEST/frames`, and each video directory contains all images of the video. Similar file structures can be seen in `data/trackingnet/TRAIN_{*}/frames`.

There are 2 JSON files in `data/trackingnet/annotations`:

`trackingnet_test.json`:  JSON file containing the annotations information of the testing set in TrackingNet dataset.
`trackingnet_train.json`:  JSON file containing the annotations information of the training set in TrackingNet dataset.

There are 2 TEXT files in `data/trackingnet/annotations`:

`trackingnet_test_infos.txt`:  TEXT file containing the information of the testing set in TrackingNet dataset.
`trackingnet_train_infos.txt`:  TEXT file containing the information of the training set in TrackingNet dataset.

#### The folder of data and annotations in OTB100

There are 98 video directories of OTB100 dataset in `data/otb100/data`, and the `img` folder under each video directory contains all images of the video.

There are only 1 JSON files in `data/otb100/annotations`:

`otb100.json`:  JSON file containing the annotations information of the OTB100 dataset.

There are only 1 TEXT files in `data/otb100/annotations`:

`otb100_infos.txt`:  TEXT file containing the information of the OTB100 dataset.

#### The folder of frames and annotations in GOT10k

There are training video directories in `data/got10k/train`, and each video directory contains all images of the video. Similar file structures can be seen in `data/got10k/test` and `data/got10k/val`.

There are 3 JSON files in `data/got10k/annotations`:

`got10k_train.json`:  JSON file containing the annotations information of the training set in GOT10k dataset.

`got10k_test.json`:  JSON file containing the annotations information of the testing set in GOT10k dataset.

`got10k_val.json`:  JSON file containing the annotations information of the valuation set in GOT10k dataset.

There are 5 TEXT files in `data/got10k/annotations`:

`got10k_train_infos.txt`:  TEXT file containing the information of the training set in GOT10k dataset.

`got10k_test_infos.txt`:  TEXT file containing the information of the testing set in GOT10k dataset.

`got10k_val_infos.txt`:  TEXT file containing the information of the valuation set in GOT10k dataset.

`got10k_train_vot_infos.txt`:  TEXT file containing the information of the `train_vot` split in GOT10k dataset.

`got10k_val_vot_infos.txt`:  TEXT file containing the information of the `val_vot` split in GOT10k dataset.

#### The folder of data and annotations in VOT2018

There are 60 video directories of VOT2018 dataset in `data/vot2018/data`, and the `color` folder under each video directory contains all images of the video.

There are only 1 JSON files in `data/vot2018/annotations`:

`vot2018.json`:  JSON file containing the annotations information of the VOT2018 dataset.

There are only 1 TEXT files in `data/vot2018/annotations`:

`vot2018_infos.txt`:  TEXT file containing the information of the VOT2018 dataset.

#### The folder of annotations in youtube_vis_2019/youtube_vis2021

There are 3 JSON files in `data/youtube_vis_2019/annotations` or `data/youtube_vis_2021/annotations`:

`youtube_vis_2019_train.json`/`youtube_vis_2021_train.json`: JSON file containing the annotations information of the training set in youtube_vis_2019/youtube_vis2021 dataset.

`youtube_vis_2019_valid.json`/`youtube_vis_2021_valid.json`: JSON file containing the annotations information of the validation set in youtube_vis_2019/youtube_vis2021 dataset.

`youtube_vis_2019_test.json`/`youtube_vis_2021_test.json`: JSON file containing the annotations information of the testing set in youtube_vis_2019/youtube_vis2021 dataset.
