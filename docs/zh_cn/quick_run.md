## 运行现有的数据集与模型

MMTracking 为现有基准测试提供了多种算法。
这些算法和基准测试在 [model_zoo.md](https://mmtracking.readthedocs.io/zh_CN/latest/model_zoo.html)中有详细说明。
以下将展示如何在现有模型和标准数据集上执行常见任务，包括：

- 使用已有模型对给定的视频或者图像文件夹进行推理。
- 在标准数据集上对已有模型进行测试（推理和评估）。
- 在标准数据集上进行训练。

### 推理

我们提供了对给定的视频或者包含连续图像的文件夹进行推理的演示脚本。
源代码可在[这里](https://github.com/open-mmlab/mmtracking/tree/master/demo/)得到。

请注意，如果您使用文件夹作为输入，则该文件夹中的图像名称必须是 **可排序的**，这意味着我们可以根据文件名中包含的数字来重新排序图像。目前，我们只支持读取文件名以'.jpg', '.jpeg' 和 '.png'结尾的图片。

#### 使用 VID 模型进行推理

以下脚本可以使用视频目标检测模型对一个输入视频进行推理。

```
python demo/demo_vid.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --checkpoint ${CHECKPOINT_FILE} \
    [--output ${OUTPUT}] \
    [--device ${DEVICE}] \
    [--show]
```

`INPUT` 和 `OUTPUT` 支持 mp4 视频格式和文件夹格式。

可选参数：

- `OUTPUT`：可视化演示的输出路径。如果未指定 `OUTPUT`，使用 `--show` 会实时显示视频。
- `DEVICE`：推理设备。可选 `cpu` 或者 `cuda:0` 等。
- `--show`：是否实时显示视频。

例子：

```shell
python ./demo/demo_vid.py \
    ./configs/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py \
    --input ${VIDEO_FILE} \
    --checkpoint ../mmtrack_output/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth \
    --output ${OUTPUT} \
    --show
```

#### 使用 MOT/VIS 模型进行推理

以下脚本可以使用多目标跟踪模型或者视频个例分割模型对单个输入视频或者图像进行推理。

```shell
python demo/demo_mot_vis.py \
    ${CONFIG_FILE} \
    --input ${INPUT} \
    [--output ${OUTPUT}] \
    [--checkpoint ${CHECKPOINT_FILE}] \
    [--score-thr ${SCORE_THR} \
    [--device ${DEVICE}] \
    [--backend ${BACKEND}] \
    [--show]
```

`INPUT` 和 `OUTPUT` 支持 mp4 视频格式和文件夹格式。

可选参数：

- `OUTPUT`：可视化演示的输出路径。如果未指定 `OUTPUT`，使用 `--show` 会实时显示视频。
- `CHECKPOINT_FILE`：如果已经在配置文件里使用 `pretrains` 关键字设置了预训练模型，那么模型权重文件是可选的。
- `SCORE_THR`: 用于过滤跟踪框的得分阈值。
- `DEVICE`：推理设备。可选 `cpu` 或者 `cuda:0` 等。
- `BACKEND`：可视化坐标框的后端。可选 `cv2` 或者 `plt`。
- `--show`：是否实时显示视频。

MOT 的例子：

```shell
python demo/demo_mot_vis.py \
    configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py \
    --input demo/demo.mp4 \
    --output mot.mp4 \
```

**注意**：当运行 `demo_mot_vis.py` 时， 我们建议您使用包含 `private` 的配置文件，因为这些配置文件不需要外部的检测结果。

VIS 的例子:

假设你已经将预训练权重下载在了 `checkpoints/` 文件夹下。

```shell
python demo/demo_mot_vis.py \
    configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
    --output ${OUTPUT} \
    --show
```

#### 使用 SOT 模型进行推理

以下脚本可以使用单目标跟踪模型对单个输入视频进行推理。

```shell
python demo/demo_sot.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --checkpoint ${CHECKPOINT_FILE} \
    [--output ${OUTPUT}] \
    [--device ${DEVICE}] \
    [--show]
```

`INPUT` 和 `OUTPUT` 支持 mp4 视频格式和文件夹格式。

可选参数：

- `OUTPUT`：可视化演示的输出路径。如果未指定 `OUTPUT`，使用 `--show` 会实时显示视频。
- `DEVICE`：推理设备。可选 `cpu` 或者 `cuda:0` 等。
- `--show`：是否实时显示视频。
- `--gt_bbox_file`: 视频的真实标注文件路径。我们只使用视频的第一帧标注。如果该参数没指定，你需要手动的画出视频初始框。

例子：

```shell
python ./demo/demo_sot.py \
    ./configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py \
    --input ${VIDEO_FILE} \
    --checkpoint ../mmtrack_output/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
    [--output ${OUTPUT}] \
    [--show] \
    [--gt_bbox_file ${GT_BBOX_FILE}]
```

### 测试

本节将展示如何在支持的数据集上测试现有模型。
支持以下测试环境：

- 单卡 GPU
- 单节点多卡 GPU
- 多节点

在测试过程中，不同的任务共享相同的 API，我们只支持 `samples_per_gpu = 1`。

您可以使用以下命令来测试一个数据集。

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

可选参数:

- `CHECKPOINT_FILE`：模型权重文件名。在应用某些 MOT 算法时不需要定义它，而是在配置文件中指定模型权重文件名。
- `RESULT_FILE`：pickle 格式的输出结果的文件名，如果不专门指定，结果将不会被专门保存成文件。
- `EVAL_METRICS`：用于评估结果的指标。允许的值取决于数据集，例如, `bbox` 适用于 ImageNet VID, `track` 适用于 LaSOT, `bbox` and `track` 都适用于 MOT17。
- `--cfg-options`：如果指定，可选配置的键值对将被合并进配置文件中。
- `--eval-options`：如果指定，可选评估配置的键值对将作为 dataset.evaluate() 函数的参数，此参数只适用于评估。
- `--format-only`：如果指定，结果将被格式化为官方格式。

#### 测试 VID 模型示例

假设您已经下载模型权重文件至文件夹 `checkpoints/` 里。

1. 在 ImageNet VID 上测试 DFF，并且评估 bbox mAP。

   ```shell
   python tools/test.py configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```

2. 使用 8 GPUs 测试 DFF，并且评估 bbox mAP。

   ```shell
   ./tools/dist_test.sh configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py 8 \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```

#### 测试 MOT 模型示例

1. 在 MOT17 上测试 Tracktor，并且评估 CLEAR MOT 指标。

   ```shell
   python tools/test.py configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py \
       --eval track
   ```

2. 使用 8 GPUs 测试 Tracktor，并且评估 CLEAR MOT 指标。

   ```shell
   ./tools/dist_test.sh  \
       configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py 8 \
       --eval track
   ```

3. 如果想使用自定义的检测器和重识别模型来测试 Trackor，你需要在 config 中更改相应的（关键字-值）对，如下所示：

```python
model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/path/to/detector_model')),
    reid=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/path/to/reid_model'))
    )
```

#### 测试 SOT 模型示例

1. 在 LaSOT 上测试 SiameseRPN++，并且评估 success 和 normed precision。

   ```shell
   python tools/test.py configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py \
       --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
       --out results.pkl \
       --eval track
   ```

2. 使用 8 GPUs 测试 SiameseRPN++，并且评估 success 和 normed precision。

   ```shell
   ./tools/dist_test.sh configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py 8 \
       --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
       --out results.pkl \
       --eval track
   ```

#### 测试 VIS 模型示例

假设你已经将预训练权重下载在了 `checkpoints/` 文件夹下。

1. 在 YouTube-VIS 2019 上测试 MaskTrack R-CNN，并且生成一个用于提交结果的 zip 文件。

   ```shell
   python tools/test.py \
       configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py \
       --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
       --out ${RESULTS_PATH}/results.pkl \
       --format-only \
       --eval-options resfile_path=${RESULTS_PATH}
   ```

2. 使用 8 GPUs 在 YouTube-VIS 2019 上测试 MaskTrack R-CNN，并且生成一个用于提交结果的 zip 文件。

   ```shell
   ./tools/dist_test.sh \
       configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py \
       --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
       --out ${RESULTS_PATH}/results.pkl \
       --format-only \
       --eval-options resfile_path=${RESULTS_PATH}
   ```

### 训练

MMTracking 也为训练模型提供了开箱即用的工具。
本节将展示如何在标准数据集（即 MOT17）上训练 _预定义_ 模型（在 [configs](../configs) 下）。

默认情况下，我们在每个 epoch 之后在验证集上评估模型，您可以通过在训练配置文件中添加 interval 参数来更改评估间隔。

```python
evaluation = dict(interval=12)  # 每 12 个 epoch 评估一次模型。
```

**重要提醒**：配置文件中的默认学习率是针对使用 8 个 GPU 的。
根据 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)，如果您使用不同数量的 GPU 或每个 GPU 使用不同数量的图片，则需要设置与 batch size 成正比的学习率，例如：`lr=0.01` 用于 8 个 GPU * 1 img/gpu， `lr=0.04` 用于 16 个 GPU * 2 imgs/gpu。

#### 单 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

在训练期间，日志文件和模型权重文件将保存到工作目录中，该目录由配置文件中的 `work_dir` 或通过 CLI 的 `--work-dir` 参数指定。

#### 多 GPU 训练

我们提供了 `tools/dist_train.sh` 来在多个 GPU 上启动训练。
基本用法如下：

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

可选参数与上述相同。

如果您想在一台机器上启动多个任务，例如，在有 8 个 GPU 的机器上进行 2 个 4-GPU 训练的任务，
您需要为每个任务指定不同的端口（默认为 29500）以避免通信冲突。

如果您使用 `dist_train.sh` 来启动训练任务，您可以使用以下命令来设置端口：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

#### 多节点训练

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

如果您使用的是 slurm 来管理多台机器，您可以使用同在单台机器上一样的命令来启动任务，但是您必须得设置合适的环境变量和参数，具体可以参考[slurm_train.sh](https://github.com/open-mmlab/mmtracking/blob/master/tools/slurm_train.sh)。

#### 使用 Slurm 管理任务

[Slurm](https://slurm.schedmd.com/) 是一款优秀的计算集群任务调度系统。
在 Slurm 管理的集群上，您可以使用 `slurm_train.sh` 来启动训练任务，并且它同时支持单节点和多节点训练。

基本用法如下：

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

您可以查看[源代码](https://github.com/open-mmlab/mmtracking/blob/master/tools/slurm_train.sh)以了解全部的参数和环境变量。

使用 Slurm 时，需要通过以下方式之一设置端口选项：

1. 通过 `--options` 设置端口。推荐使用此方式，因为它不会更改原始配置。

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
   ```

2. 修改配置文件以设置不同的通信端口。

   在 `config1.py` 中设置

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   在 `config2.py` 中设置

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   然后可以使用 `config1.py` 和 `config2.py` 启动两个任务。

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

#### 训练 VID 模型示例

1. 在 ImageNet VID 和 ImageNet DET 上 训练 DFF，接着在最后一个 epoch 评估 bbox mAP.

```shell
bash ./tools/dist_train.sh ./configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py 8 --work-dir ./work_dirs/
```

#### 训练 MOT 模型示例

对于像 MOT、SORT、DeepSORT 以及 Trackor 这样的 MOT 方法，你需要训练一个检测器和一个 reid 模型，而非直接训练 MOT 模型。

1. 训练检测器

   如果你想要为多目标跟踪器训练检测器，为了兼容 MMDetection, 你只需要在 config 里面增加一行代码 `USE_MMDET=True`, 然后使用与 MMDetection 相同的方式运行它。可参考示例  [faster_rcnn_r50_fpn.py](https://github.com/open-mmlab/mmtracking/blob/master/configs/_base_/models/faster_rcnn_r50_fpn.py)。

   请注意 MMTracking 和 MMDetection 在 base config 上有些许不同：`detector` 仅仅是 `model` 的一个子模块。例如，MMDetection 中的 Faster R-CNN 的 config如下：

   ```python
       model = dict(
           type='FasterRCNN',
           ...
       )
   ```

   但在 MMTracking 中，config 如下：

   ```python
   model = dict(
       detector=dict(
           type='FasterRCNN',
           ...
       )
   )
   ```

   这里有一个在 MOT17 上训练检测器模型，并在每个 epoch 结束后评估 bbox mAP 的范例：

   ```shell
   bash ./tools/dist_train.sh ./configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py 8 \
       --work-dir ./work_dirs/
   ```

2. 训练 ReID 模型

   你可能需要在 MOT 或其它实际应用中训练 ReID 模型。我们在 MMTracking 中也支持 ReID 模型的训练，这是基于 [MMClassification](https://github.com/open-mmlab/mmclassification) 实现的。

   这里有一个在 MOT17 上训练检测器模型，并在每个 epoch 结束后评估 bbox mAP 的范例：

   ```shell
   bash ./tools/dist_train.sh ./configs/reid/resnet50_b32x8_MOT17.py 8 \
       --work-dir ./work_dirs/
   ```

3. 完成检测器和 ReID 模型训练后，可参考[测试MOT模型示例](https://mmtracking.readthedocs.io/zh_CN/latest/quick_run.html#mot)来测试多目标跟踪器。

#### 训练 SOT 模型示例

1. 在 COCO、ImageNet VID 和 ImageNet DET 上训练 SiameseRPN++，然后从第 10 个 epoch 到第 20 个 epoch 评估其 success、precision 和 normed precision。

   ```shell
   bash ./tools/dist_train.sh ./configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py 8 \
       --work-dir ./work_dirs/
   ```

#### 训练 VIS 模型示例

1. 在 YouTube-VIS 2019 数据集上训练 MaskTrack R-CNN。由于 YouTube-VIS 没有提供 validation 集的注释文件，因此在训练过程中不会进行评估。

   ```shell
   bash ./tools/dist_train.sh ./configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py 8 \
       --work-dir ./work_dirs/
   ```

## 使用自定义数据集和模型运行

在本节中，您将了解如何使用自定义数据集和模型进行推理、测试和训练：

基本步骤如下：

1. 准备自定义数据集（如果适用）
2. 准备自定义模型（如果适用）
3. 准备配置文件
4. 训练新模型
5. 测试、推理新模型

### 1. 准备自定义数据集

在 MMTracking 中支持新数据集的方式有以下两种：

1. 将数据集重组为 CocoVID 格式。
2. 实现一个新的数据集。

通常我们建议使用第一种方法，它比第二种方法容易实现。

[tutorials/customize_dataset.md](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html) 中提供了有关自定义数据集的详细教程。

### 2. 准备自定义模型

我们提供了不同任务下自定义模型的教程：

- [tutorials/customize_mot_model.md](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_vid_model.html)
- [tutorials/customize_sot_model.md](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_mot_model.html)
- [tutorials/customize_vid_model.md](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_sot_model.html)

### 3. 准备配置文件

下一步是准备配置文件，从而可以成功加载数据集或模型。
[tutorials/config.md](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config.html) 提供了有关配置系统的更多详细教程。

### 4. 训练新模型

要使用新的配置文件训练模型，您只需运行

```shell
python tools/train.py ${NEW_CONFIG_FILE}
```

更详细的用法请参考前面的训练说明。

### 5. 测试和推理

要测试经过训练的模型，您只需运行

```shell
python tools/test.py ${NEW_CONFIG_FILE} ${TRAINED_MODEL} --eval bbox track
```

更详细的用法请参考前面的测试或推理说明。
