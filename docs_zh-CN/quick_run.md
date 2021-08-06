## 运行现有的数据集与模型

MMTracking 为现有基准测试提供了多种算法。
这些算法和基准测试的详情分别在 [model_zoo.md](model_zoo.md) 和 [dataset.md](dataset.md) 中说明。
以下将展示如何在现有模型和标准数据集上执行常见任务，包括：

- 使用已有模型对给定的视频或者图像文件夹进行推理。
- 在标准数据集上对已有模型进行测试（推理和评估）。
- 在标准数据集上进行训练。

### 推理

我们提供了对给定的视频或者包含连续图像的文件夹进行推理的演示脚本。
源代码可在[这里](../demo/)得到。

请注意，如果您使用文件夹作为输入，则该文件夹中应该只有图像，并且图像名称必须是 **可排序的**，这意味着我们可以根据文件名重新排序图像。

### 使用 MOT 模型进行推理

以下脚本可以使用多目标跟踪模型对一个输入视频或者图像进行推理。

```shell
python demo/demo_mot.py \
    ${CONFIG_FILE} \
    --input ${INPUT} \
    [--output ${OUTPUT}] \
    [--checkpoint ${CHECKPOINT_FILE}] \
    [--device ${DEVICE}] \
    [--backend ${BACKEND}] \
    [--show]
```

`INPUT` 和 `OUTPUT` 支持 mp4 视频格式和文件夹格式。

可选参数：

- `OUTPUT`：可视化演示的输出路径。如果未指定 `OUTPUT`，使用 `--show` 会实时显示视频。
- `CHECKPOINT_FILE`：如果已经在配置文件里使用 `pretrains` 关键字设置了预训练模型，那么模型权重文件是可选的。
- `DEVICE`：推理设备。可选 `cpu` 或者 `cuda:0` 等。
- `BACKEND`：可视化坐标框的后端。可选 `cv2` 或者 `plt`。
- `--show`：是否实时显示视频。

例子：

```shell
python demo/demo_mot.py configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py --input demo/demo.mp4 --output mot.mp4
```

注意：当运行`demo_mot.py`时， 我们建议您使用包含`private`的配置文件，这是因为这些配置文件不需要外部的检测结果。

### 使用 SOT 模型进行推理

以下脚本可以使用单目标跟踪模型对一个输入视频进行推理。

```shell
python demo/demo_sot.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --checkpoint ${CHECKPOINT_FILE} \
    [--output ${OUTPUT}] \
    [--device ${DEVICE}] \
    [--show]
```

可选参数：

- `OUTPUT`：可视化演示的输出路径。如果未指定 `OUTPUT`，使用 `--show` 会实时显示视频。
- `DEVICE`：推理设备。可选 `cpu` 或者 `cuda:0` 等。
- `--show`：是否实时显示视频。

例子：

```shell
python ./demo/demo_sot.py \
    ./configs/sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py \
    --input ${VIDEO_FILE} \
    --checkpoint ../mmtrack_output/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth \
    --output ${OUTPUT} \
    --show
```

### 使用 VID 模型进行推理

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

例子：

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

3. 在 LaSOT 上测试 SiameseRPN++，并且评估 success 和 normed precision。

   ```shell
   python tools/test.py configs/sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py \
       --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth \
       --out results.pkl \
       --eval track
   ```

4. 使用 8 GPUs 测试 SiameseRPN++，并且评估 success 和 normed precision。

   ```shell
   ./tools/dist_test.sh configs/sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py 8 \
       --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth \
       --out results.pkl \
       --eval track
   ```

5. 在 MOT17 上测试 Tracktor，并且评估 CLEAR MOT 指标。

   ```shell
   python tools/test.py configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py \
       --eval track
   ```

6. 使用 8 GPUs 测试 Tracktor，并且评估 CLEAR MOT 指标。

   ```shell
   ./tools/dist_test.sh  \
       configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py 8 \
       --eval track
   ```

### 训练

MMTracking 也为训练模型提供了开箱即用的工具。
本节将展示如何在标准数据集（即 MOT17）上训练 _预定义_ 模型（在 [configs](../configs) 下）。

默认情况下，我们在每个 epoch 之后在验证集上评估模型，您可以通过在训练配置文件中添加 interval 参数来更改评估间隔。

```python
evaluation = dict(interval=12)  # 每 12 个 epoch 评估一次模型。
```

**重要提醒**：配置文件中的默认学习率是针对使用 8 个 GPU 的。
根据 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)，如果您使用不同数量的 GPU 或每个 GPU 使用不同数量的图片，则需要设置与 batch size 成正比的学习率，例如 `lr=0.01` 用于 8 个 GPU \* 1 img/gpu 和 `lr=0.04` 用于 16 个 GPU \* 2 imgs/gpu。

### 单 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

在训练期间，日志文件和模型权重文件将保存到工作目录中，该目录由配置文件中的 `work_dir` 或通过 CLI 的 `--work-dir` 参数指定。

### 多 GPU 训练

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

### 多节点训练

MMTracking 依赖 `torch.distributed` 包进行分布式训练。
因此，作为一种基本用法，可以通过 PyTorch 的 [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility) 启动分布式训练。

### 使用 Slurm 管理任务

[Slurm](https://slurm.schedmd.com/) 是一款优秀的计算集群任务调度系统。
在 Slurm 管理的集群上，您可以使用 `slurm_train.sh` 来启动训练任务，并且它同时支持单节点和多节点训练。

基本用法如下：

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

您可以查看[源代码](../tools/slurm_train.sh)以了解完整的参数和环境变量。

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

## 使用自定义数据集和模型运行

在本节中，您将了解如何使用自定义数据集和模型进行推理、测试和训练：

基本步骤如下：

1. 准备自定义数据集（如果适用）
2. 准备自定义模型（如果适用）
3. 准备配置文件
4. 训练、测试、推理新模型

### 准备自定义数据集

在 MMTracking 中支持新数据集的方式有以下两种：

1. 将数据集重组为 CocoVID 格式。
2. 实现一个新的数据集。

通常我们建议使用第一种方法，它比第二种方法容易实现。

[tutorials/customize_dataset.md](tutorials/customize_dataset.md) 中提供了有关自定义数据集的详细教程。

### 准备自定义模型

我们提供了不同任务下自定义模型的教程：

- [tutorials/customize_mot_model.md](tutorials/customize_mot_model.md)
- [tutorials/customize_sot_model.md](tutorials/customize_sot_model.md)
- [tutorials/customize_vid_model.md](tutorials/customize_vid_model.md)

### 准备配置文件

下一步是准备配置文件，从而可以成功加载数据集或模型。
[tutorials/config.md](tutorials/config.md) 提供了有关配置系统的更多详细教程。

### 训练新模型

要使用新的配置文件训练模型，您只需运行

```shell
python tools/train.py ${NEW_CONFIG_FILE}
```

更详细的用法请参考上面的训练说明。

### 测试和推理

要测试经过训练的模型，您只需运行

```shell
python tools/test.py ${NEW_CONFIG_FILE} ${TRAINED_MODEL} --eval bbox track
```

更详细的用法请参考上面的测试或推理说明。
