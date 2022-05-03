## 数据集准备

本页提供关于现有基准测试的数据集准备的说明，包括：

- 视频目标检测
  - [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)
- 多目标跟踪
  - [MOT Challenge](https://motchallenge.net/)
  - [CrowdHuman](https://www.crowdhuman.org/)
  - [LVIS](https://www.lvisdataset.org/)
  - [TAO](https://taodataset.org/)
  - [DanceTrack](https://dancetrack.github.io)
- 单目标跟踪
  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
  - [UAV123](https://cemse.kaust.edu.sa/ivul/uav123/)
  - [TrackingNet](https://tracking-net.org/)
  - [OTB100](http://www.visual-tracking.net/)
  - [GOT10k](http://got-10k.aitestunion.com/)
  - [VOT2018](https://www.votchallenge.net/vot2018/)
- 视频实例分割
  - [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

### 1. 下载数据集

请从官方网站下载数据集。建议将数据集的根目录符号链接到 `$MMTRACKING/data`。

#### 1.1 视频目标检测

- 对于视频目标检测任务的训练和测试，只需要 ILSVRC 数据集。

- `ILSVRC` 下的 `Lists` 包含来自在[这里](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets)的 txt 文件。

#### 1.2 多目标跟踪

- 对于多目标跟踪任务的训练和测试，需要 MOT Challenge 中的任意一个数据集（比如 MOT17, TAO和DanceTrack)， CrowdHuman 和 LVIS 可以作为补充数据。

- `tao` 文件夹下包含官方标注的 `annotations` 可以从[这里](https://github.com/TAO-Dataset/annotations)获取。

- `lvis` 文件夹下包含 lvis-v0.5 官方标注的 `annotations` 可以从[这里](https://github.com/lvis-dataset/lvis-api/issues/23#issuecomment-894963957)下载。`./tools/convert_datasets/tao/merge_coco_with_lvis.py` 脚本中需到的同义词映射文件 `coco_to_lvis_synset.json` 可以从[这里](https://github.com/TAO-Dataset/tao/tree/master/data)获取。

#### 1.3 单目标跟踪

- 对于单目标跟踪任务的训练和测试，需要 MSCOCO， ILSVRC, LaSOT, UAV123, TrackingNet, OTB100 和 GOT10k 数据集。

- 对于 OTB100 数据集，你不必要手工地从官网下载数据。我们提供了下载脚本。

```shell
# 通过网页爬虫下载 OTB100 数据集
python ./tools/convert_datasets/otb100/download_otb100.py -o ./data/otb100/zips -p 8
```

- 对于 VOT2018, 我们使用官方的下载脚本。

```shell
# 通过网页爬虫下载 VOT2018 数据集
python ./tools/convert_datasets/vot/download_vot.py --dataset vot2018 --save_path ./data/vot2018/data
```

#### 1.4 视频实例分割

- 对于视频实例分割任务的训练和测试，只需要 YouTube-VIS 中的任意一个数据集（比如 YouTube-VIS 2019）。

#### 1.5 数据集文件夹结构

如果您的文件夹结构与以下不同，您可能需要更改配置文件中的相应路径。

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

### 2. 转换标注格式

我们使用 [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) 来维护代码库中所有的数据集。

基于此，您需要将官方的标注转换为此种格式。我们提供的脚本以及用法如下：

```shell
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# MOT17
# MOT Challenge中其余数据集的处理与MOT17相同
python ./tools/convert_datasets/mot/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/convert_datasets/mot/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3

# DanceTrack
python ./tools/convert_datasets/dancetrack/dancetrack2coco.py -i ./data/DanceTrack ./data/DanceTrack/annotations

# CrowdHuman
python ./tools/convert_datasets/mot/crowdhuman2coco.py -i ./data/crowdhuman -o ./data/crowdhuman/annotations

# LVIS
# 合并 LVIS 和 COCO 的标注来训练 QDTrack
python ./tools/convert_datasets/tao/merge_coco_with_lvis.py --lvis ./data/lvis/annotations/lvis_v0.5_train.json --coco ./data/coco/annotations/instances_train2017.json --mapping ./data/lvis/annotations/coco_to_lvis_synset.json --output-json ./data/lvis/annotations/lvisv0.5+coco_train.json

# TAO
# 为 QDTrack 生成过滤后的json文件
python ./tools/convert_datasets/tao/tao2coco.py -i ./data/tao/annotations --filter-classes

# LaSOT
python ./tools/convert_datasets/lasot/gen_lasot_infos.py -i ./data/lasot/LaSOTBenchmark -o ./data/lasot/annotations

# UAV123
# 下载标注
# 由于UAV123数据集的所有视频的标注信息不具有统一性，我们仅需下载提前生成的数据信息文件即可。
wget https://download.openmmlab.com/mmtracking/data/uav123_infos.txt -P data/uav123/annotations

# TrackingNet
# 解压目录 'data/trackingnet/' 下的所有 '*.zip' 文件
bash ./tools/convert_datasets/trackingnet/unzip_trackingnet.sh ./data/trackingnet
# 生成标注
python ./tools/convert_datasets/trackingnet/gen_trackingnet_infos.py -i ./data/trackingnet -o ./data/trackingnet/annotations

# OTB100
# 解压目录 'data/otb100/zips' 下的所有 '*.zip' 文件
bash ./tools/convert_datasets/otb100/unzip_otb100.sh ./data/otb100
# 下载标注
# 由于UAV123数据集的所有视频的标注信息不具有统一性，我们仅需下载提前生成的数据信息文件即可。
wget https://download.openmmlab.com/mmtracking/data/otb100_infos.txt -P data/otb100/annotations

# GOT10k
# 解压 'data/got10k/full_data/test_data.zip', 'data/got10k/full_data/val_data.zip' 和 目录'data/got10k/full_data/train_data/' 下的所有 '*.zip' 文件
bash ./tools/convert_datasets/got10k/unzip_got10k.sh ./data/got10k
# 生成标注
python ./tools/convert_datasets/got10k/gen_got10k_infos.py -i ./data/got10k -o ./data/got10k/annotations

# VOT2018
python ./tools/convert_datasets/vot/gen_vot_infos.py -i ./data/vot2018 -o ./data/vot2018/annotations --dataset_type vot2018

# YouTube-VIS 2019
python ./tools/convert_datasets/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2019 -o ./data/youtube_vis_2019/annotations --version 2019

# YouTube-VIS 2021
python ./tools/convert_datasets/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2021 -o ./data/youtube_vis_2021/annotations --version 2021
```

完成以上格式转换后，文件目录结构如下：

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

#### ILSVRC 的标注文件夹

在`data/ILSVRC/annotations`中有 3 个 JSON 文件:

`imagenet_det_30plus1cls.json`: 包含 ImageNet DET 训练集标注信息的json文件。`30plus1cls` 中的 `30` 表示本数据集与 ImageNet VID 数据集重合的30类，`1cls` 表示我们将 ImageNet Det 数据集中的其余170类作为一类，
并命名为 `other_categeries`。

`imagenet_vid_train.json`: 包含 ImageNet VID 训练集标注信息的 JSON 文件。

`imagenet_vid_val.json`: 包含 ImageNet VID 验证集标注信息的 JSON 文件。

#### MOT15/MOT16/MOT17/MOT20 的标注和 reid 文件夹

我们以MOT17为例，其余数据集结构相似。

在 `data/MOT17/annotations` 中有 8 个 JSON 文件:

`train_cocoformat.json`: 包含 MOT17 训练集标注信息的 JSON 文件。

`train_detections.pkl`: 包含 MOT17 训练集公共检测结果信息的 pickle 文件。

`test_cocoformat.json`: 包含 MOT17 测试集标注信息的 JSON 文件。

`test_detections.pkl`: 包含 MOT17 测试集公共检测结果信息的 pickle 文件。

`half-train_cocoformat.json`, `half-train_detections.pkl`, `half-val_cocoformat.json` 以及 `half-val_detections.pkl` 具有和 `train_cocoformat.json`、`train_detections.pkl` 相似的含义。 `half` 意味着我们将训练集中的每个视频分成两半。 前一半标记为 `half-train`, 后一半标记为 `half-val`。

`data/MOT17/reid` 的目录结构如下:

```

reid
├── imgs
│   ├── MOT17-02-FRCNN_000002
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ......
│   ├── MOT17-02-FRCNN_000003
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── ......
├── meta
│   ├── train_80.txt
│   ├── val_20.txt

```

`train_80.txt` 中的 `80` 意味着将全部 ReID 数据集的80%作为训练集，剩余的20%作为验证集。

训练集标注 `train_80.txt` 中每一行包含一个文件名和其对应的图片物体真实标签。格式如下：

```

MOT17-05-FRCNN_000110/000018.jpg 0
MOT17-13-FRCNN_000146/000014.jpg 1
MOT17-05-FRCNN_000088/000004.jpg 2
MOT17-02-FRCNN_000009/000081.jpg 3

```

`MOT17-05-FRCNN_000110` 表示 `MOT17-05-FRCNN` 视频中的第110个人。

验证集标注 `val_20.txt` 的结构和上面类似。

`reid/imgs` 中的图片是从 `MOT17/train` 中原始图片根据对应的 `gt.txt` 裁剪得到。真实类别标签值在 `[0, num_classes - 1]` 范围内。

#### crowdhuman 的标注文件夹

在 `data/crowdhuman/annotations` 中有 2 个 JSON 文件:

`crowdhuman_train.json`:  包含 CrowdHuman 训练集标注信息的 JSON 文件。
`crowdhuman_val.json`:  包含 CrowdHuman 验证集标注信息的 JSON 文件。

#### lvis 的标注文件夹

在`data/lvis/annotations` 中有 8 个 JSON 文件:

`coco_to_lvis_synset.json`: 包含 COCO 和 LVIS 类别映射关系的 JSON 文件。

`lvisv0.5+coco_train.json`: 包含合并后标注的 JSON 文件。

`lvis_v0.5_train.json`: 包含 lvisv0.5 训练集标注信息的 JSON 文件。

`lvis_v0.5_val.json`: 包含 lvisv0.5 测试集标注信息的 JSON 文件。

`lvis_v1_train.json`: 包含 lvisv1 训练集标注信息的 JSON 文件。

`lvis_v1_val.json`: 包含 lvisv1 测试集标注信息的 JSON 文件。

`lvis_v1_image_info_test_challenge.json`: 包含可全年使用的 lvisv1 测试集标注 JSON 文件。

`lvis_v1_image_info_test_dev.json`: 包含仅一年一次供 LVIS Challenge 使用的 lvisv1 测试集标注 JSON 文件。

#### tao 的标注文件夹

在`data/tao/annotations` 中有 9 个 JSON 文件:

`test_categories.json`: 包含在 TAO 测试集中会被评估的类别序列的 JSON 文件。

`test_without_annotations.json`:  包含测试视频的 JSON 文件。 `images` 和 `videos` 域包含会在测试集中被评估的图片和视频。

`test_482_classes.json`: 包含测试集转换结果的 JSON 文件。

`train.json`: 包含 TAO 训练集中 LVIS 类别标注的 JSON 文件。

`train_482_classes.json`: 包含训练集转换结果的 JSON 文件。

`train_with_freeform.json`: 包含 TAO 训练集所有类别标注的 JSON 文件。

`validation.json`: 包含 TAO 验证集中 LVIS 类别标注的 JSON 文件。

`validation_482_classes.json`: 包含验证集转换结果的 JSON 文件。

`validation_with_freeform.json`: 包含 TAO 验证集所有类别标注的 JSON 文件。

#### lasot 的标注文件夹

在 `data/lasot/annotations` 中有 2 个 JSON 文件:

`lasot_train.json`:  包含 LaSOT 训练集标注信息的 JSON 文件。
`lasot_test.json`:  包含 LaSOT 测试集标注信息的 JSON 文件。

在 `data/lasot/annotations` 中有 2 个 TEXT 文件:

`lasot_train_infos.txt`:  包含 LaSOT 训练集信息的 TEXT 文件。
`lasot_test_infos.txt`:  包含 LaSOT 测试集信息的 TEXT 文件。

#### UAV123 的标注文件夹

在 `data/UAV123/annotations` 中只有 1 个 JSON 文件:

`uav123.json`: 包含 UAV123 数据集标注信息的 JSON 文件。

在 `data/UAV123/annotations` 中有 1 个 TEXT 文件:

`uav123_infos.txt`:  包含 UAV123 数据集信息的 TEXT 文件。

#### TrackingNet 的标注和视频帧文件夹

在 `data/trackingnet/TEST/frames` 文件夹下有 TrackingNet 测试集的 511 个视频目录， 每个视频目录下面包含该视频所有图片。`data/trackingnet/TRAIN_{*}/frames` 下具有类似的文件目录结构。

在 `data/trackingnet/annotations` 中有 2 个 JSON 文件：

`trackingnet_train.json`： 包含 TrackingNet 训练集标注信息的 JSON 文件。
`trackingnet_test.json`： 包含 TrackingNet 测试集标注信息的 JSON 文件。

在 `data/trackingnet/annotations` 中有 2 个 TEXT 文件：

`trackingnet_train_infos.txt`： 包含 TrackingNet 训练集信息的 TEXT 文件。
`trackingnet_test_infos.txt`： 包含 TrackingNet 测试集信息的 TEXT 文件。

#### OTB100 的标注和视频帧文件夹

在 `data/otb100/data` 文件夹下有 OTB100 数据集的 98 个视频目录， 每个视频目录下的 `img` 文件夹包含该视频所有图片。

在 `data/otb100/data/annotations` 中只有 1 个 JSON 文件：

`otb100.json`： 包含 OTB100 数据集标注信息的 JSON 文件

在 `data/otb100/annotations` 中有 1 个 TEXT 文件:

`otb100_infos.txt`:  包含 OTB100 数据信息的 TEXT 文件。

#### GOT10k 的标注和视频帧文件夹

在 `data/got10k/train` 文件夹下有 GOT10k 训练集的视频目录， 每个视频目录下面包含该视频所有图片。`data/got10k/test` 和 `data/got10k/val` 下具有类似的文件目录结构。

在 `data/got10k/annotations` 中有 3 个 JSON 文件：

`got10k_train.json`： 包含 GOT10k 训练集标注信息的 JSON 文件。

`got10k_test.json`： 包含 GOT10k 测试集标注信息的 JSON 文件。

`got10k_val.json`： 包含 GOT10k 验证集标注信息的 JSON 文件。

在 `data/got10k/annotations` 中有 5 个 TEXT 文件：

`got10k_train_infos.txt`： 包含 GOT10k 训练集信息的 TEXT 文件。

`got10k_test_infos.txt`： 包含 GOT10k 测试集信息的 TEXT 文件。

`got10k_val_infos.txt`： 包含 GOT10k 验证集信息的 TEXT 文件。

`got10k_train_vot_infos.txt`： 包含 GOT10k `train_vot` 划分集信息的 TEXT 文件。

`got10k_val_vot_infos.txt`： 包含 GOT10k `val_vot` 划分集信息的 TEXT 文件。

#### VOT2018的标注和视频帧文件夹

在 `data/vot2018/data` 文件夹下有 VOT2018 数据集的 60 个视频目录， 每个视频目录下的 `color` 文件夹包含该视频所有图片。

在 `data/vot2018/data/annotations` 中只有一个 JSON 文件：

`vot2018.json`： 包含 VOT2018 数据集标注信息的 JSON 文件。

在 `data/vot2018/data/annotations` 中只有一个 TEXT 文件：

`vot2018_infos.txt`： 包含 VOT2018 数据集信息的 TEXT 文件。

#### youtube_vis_2019/youtube_vis2021 的标注文件夹

在 `data/youtube_vis_2019/annotations` 或者 `data/youtube_vis_2021/annotations` 下有 3 个 JSON 文件：

`youtube_vis_2019_train.json`/`youtube_vis_2021_train.json`: 包含着 youtube_vis_2019/youtube_vis2021 训练集注释信息的 JSON 文件。

`youtube_vis_2019_valid.json`/`youtube_vis_2021_valid.json`: 包含着 youtube_vis_2019/youtube_vis2021 验证集注释信息的 JSON 文件。

`youtube_vis_2019_test.json`/`youtube_vis_2021_test.json`: 包含着 youtube_vis_2019/youtube_vis2021 测试集注释信息的 JSON 文件。
