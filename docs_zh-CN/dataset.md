## 数据集准备

本页提供关于现有基准测试的数据集准备的说明，包括：

- 视频目标检测
  - [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)
- 多目标跟踪
  - [MOT Challenge](https://motchallenge.net/)
- 单目标跟踪
  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

### 1. 下载数据集

请从官方网站下载数据集。建议将数据集的根目录符号链接到 `$MMTRACKING/data`。如果您的文件夹结构与以下不同，您可能需要更改配置文件中的相应路径。

注意：

- `ILSVRC` 下的 `Lists` 包含来自在[这里](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets)的 txt 文件。

- 对于视频目标检测任务的训练和测试，只需要 ILSVRC 数据集。

- 对于多目标跟踪任务的训练和测试，只需要 MOT Challenge 中的任意一个数据集（比如 MOT17）。

- 对于单目标跟踪任务的训练和测试，需要 MSCOCO，ILSVRC 和 LaSOT 数据集。

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
|   │
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
|   │
│   ├── lasot
│   │   ├── LaSOTTesting
│   │   │   ├── airplane-1
│   │   │   ├── airplane-13
|   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
```

### 2. 转换标注格式

我们使用 [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) 来维护代码库中所有的数据集。

基于此，您需要将官方的标注转换为此种格式。我们提供的脚本以及用法如下：

```shell
# ImageNet DET
python ./tools/convert_datasets/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# LaSOT
python ./tools/convert_datasets/lasot2coco.py -i ./data/lasot/LaSOTTesting -o ./data/lasot/annotations

# MOT17
# MOT Challenge中其余数据集的处理与MOT17相同
python ./tools/convert_datasets/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
python ./tools/convert_datasets/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3
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
|   │
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
|   │
│   ├── lasot
│   │   ├── LaSOTTesting
│   │   │   ├── airplane-1
│   │   │   ├── airplane-13
│   │   ├── annotations
|   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   ├── test
|   |   ├── annotations
|   |   ├── reid
│   │   │   ├── imgs
│   │   │   ├── meta
```

#### ILSVRC的标注文件夹

在`data/ILSVRC/annotations`中有3个 json 文件:

`imagenet_det_30plus1cls.json`: 包含 ImageNet DET 训练集标注信息的json文件。`30plus1cls` 中的 `30` 表示本数据集与 ImageNet VID 数据集重合的30类，`1cls` 表示我们将 ImageNet Det 数据集中的其余170类作为一类，
并命名为 `other_categeries`。

`imagenet_vid_train.json`: 包含 ImageNet VID 训练集标注信息的 json 文件。

`imagenet_vid_val.json`: 包含 ImageNet VID 验证集标注信息的 json 文件。

#### lasot的标注文件夹

在 `data/lasot/annotations` 中有1个 json 文件:

`lasot_test.json`:  包含 LaSOT 测试集标注信息的 json 文件。

#### MOT15/MOT16/MOT17/MOT20的标注和reid文件夹

我们以MOT17为例，其余数据集结构相似。

在 `data/MOT17/annotations` 中有8个 json 文件:

`train_cocoformat.json`: 包含 MOT17 训练集标注信息的 json 文件。

`train_detections.pkl`: 包含 MOT17 训练集公共检测结果信息的 pickle 文件。

`test_cocoformat.json`: 包含 MOT17 测试集标注信息的 json 文件。

`test_detections.pkl`: 包含 MOT17 测试集公共检测结果信息的 pickle 文件。

`half-train_cocoformat.json`, `half-train_detections.pkl`, `half-val_cocoformat.json` 以及 `half-val_detections.pkl` 具有和 `train_cocoformat.json`、`train_detections.pkl` 相似的含义。 `half` 意味着我们将训练集中的每个视频分成两半。 前一半标记为 `half-train`, 后一半标记为 `half-val`。

`data/MOT17/reid` 的目录结构如下:

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
