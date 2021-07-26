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

- 对于多目标跟踪任务的训练和测试，只需要 MOT17 数据集。

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

### 2. 转换标注格式

我们使用 [CocoVID](../mmtrack/datasets/parsers/coco_video_parser.py) 来维护代码库中所有的数据集。

基于此，您需要将官方的标注转换为此种格式。我们提供的脚本以及用法如下：

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
