# 学习如何训练模型并测试

## 训练模型

本节将展示如何在支持的数据集上训练现有模型。

支持以下训练环境：

- CPU
- 单 GPU
- 单节点多 GPU
- 多节点

您也可以使用 Slurm 完成工作。

重点：

- 您可以通过修改 `train_cfg` 为 `train_cfg = dict(val_interval=10)` 。

  从而在训练过程中更改评估间隔。这意味着每 10 个周期对模型进行一次评估。

- 所有配置文件的学习率设置默认是基于 8 个 GPUs 训练。

  根据 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677) ，

  如果你使用不同数量的 GPUs 或者 单张 GPU 上图片数量有改动， 你需要设置学习率与批数据大小成正比。

  例如，`lr=0.01` 对应 8 GPU * 1 img/gpu，lr=0.04 对应 16 GPU * 2 imgs/gpu 。

- 在训练过程中，日志文件和历史模型会保存到工作目录中，工作目录由 CLI 参数 `--work-dir` 指定。它使用 `./work_dirs/CONFIG_NAME` 作为默认值。

- 如果你想要混合精确训练，只需指定 CLI 参数 `--amp` 。

#### 1. 基于 CPU 进行训练

模型默认放在 cuda 设备上训练。只有在没有 cuda 设备的情况下，模型才会放在 CPU 上。所以如果你想在 CPU 上训练模型，你需要设置 `export CUDA_VISIBLE_DEVICES=-1` 来禁用 GPU 可见性。更多内容详见 [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850)

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [optional arguments]
```

在 CPU 上训练 VID 模型 DFF 的例子：

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/vid/dff/dff_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py
```

#### 2. 基于单 GPU 进行训练

如果你想在单 GPU 上训练模型，你可以直接使用 `tools/train.py` 如下所示。

```shell script
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

你可以使用 `export CUDA_VISIBLE_DEVICES=$GPU_ID` 来选择 GPU 。

在单 GPU 上训练 MOT 模型 ByteTrack 的例子：

```shell script
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py
```

#### 3. 基于单节点多 GPU 进行训练

我们提供 `tools/dist_train.sh` 在多个 GPU 上训练。

基本用法如下：

```shell script
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

如果您想在一台机器上启动多个作业，例如，在一台8个 GPU 的机器上进行2个 4-GPU 训练作业，

您需要为每个作业指定不同的端口(默认为29500)，以避免通信冲突。

您可以通过以下命令设置端口：

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

在单节点多 GPU 上训练 SOT 模型 SiameseRPN++ 的例子：

```shell script
bash ./tools/dist_train.sh ./configs/sot/siamese_rpn/siamese-rpn_r50_8xb16-20e_imagenetvid-imagenetdet-coco_test-otb100.py 8
```

#### 4. 基于多节点进行训练

如果你启动多台通过以太网简单连接的机器。你可以直接运行以下命令：

在第一台机器上：

```shell script
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上：

```shell script
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

如果你没有像 InfiniBand 这样的高速网络，它通常是很慢的。

#### 5. 基于 Slurm 进行训练

[Slurm](https://slurm.schedmd.com/) 是计算集群的一个很好的作业调度系统。在由 Slurm 管理的集群上，可以使用 `slurm_train.sh` 来生成培训作业。它支持单节点和多节点训练。

基本用法如下：

```shell script
bash ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPUS}
```

用 Slurm 训练 VIS 模型 MaskTrack R-CNN 的例子：

```shell script
PORT=29501 \
GPUS_PER_NODE=8 \
SRUN_ARGS="--quotatype=reserved" \
bash ./tools/slurm_train.sh \
mypartition \
masktrack \
configs/vis/masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py \
./work_dirs/MaskTrack_RCNN \
8
```

## 测试

本节将展示如何在受支持的数据集上测试现有模型。

支持以下测试环境：

- CPU
- 单 GPU
- 单节点多 GPU
- 多节点

您也可以使用 Slurm 完成工作。

重点：

- 您可以通过修改评估器中的 `outfile_prefix` 来设置结果保存路径。例如， `val_evaluator = dict(outfile_prefix='results/stark_st1_trackingnet')` 。否则，将创建一个临时文件，并将在评估后删除。
- 如果你只想要格式化的结果而不需要求值，你可以设置 `format_only=True` 。例如， `test_evaluator = dict(type='YouTubeVISMetric', metric='youtube_vis_ap', outfile_prefix='./youtube_vis_results', format_only=True)`

#### 1. 基于 CPU 进行测试

模型默认放在 cuda 设备上训练。只有在没有 cuda 设备的情况下，模型才会放在 CPU 上。所以如果你想在 CPU 上测试模型，你需要设置 `export CUDA_VISIBLE_DEVICES=-1` 来禁用 GPU 可见性。详情请浏览 [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850) 。

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} [optional arguments]
```

在 CPU 上测试 VID 模型 DFF 的例子：

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/vid/dff/dff_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py --checkpoint https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid/dff_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_213250-548911a4.pth
```

#### 2. 基于单 GPU 进行测试

如果你想在单个 GPU 上测试模型，你可以直接使用下面的 `tools/test.py` 。

```shell script
python tools/test.py ${CONFIG_FILE} [optional arguments]
```

你可以使用 `export CUDA_VISIBLE_DEVICES=$GPU_ID` 来选择 GPU 。

在单 GPU 上测试 MOT 模型 ByteTrack 的一个例子：

```shell script
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py --checkpoint https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
```

#### 3. 基于单节点多 GPU 进行测试

我们提供 `tools/dist_test.sh` 在多个 GPU 上启动测试。基本用法如下。

```shell script
bash ./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

在单节点多 GPU 上测试 SOT 模型 siameserpn++ 的例子：

```shell script
bash ./tools/dist_test.sh ./configs/sot/siamese_rpn/siamese-rpn_r50_8xb16-20e_imagenetvid-imagenetdet-coco_test-otb100.py 8 --checkpoint https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_20e_otb100_20220421_144232-6b8f1730.pth
```

#### 4. 基于多节点进行测试

您可以在多个节点上进行测试，这与 ”在多个节点上进行训练” 类似。

#### 5. 基于 Slurm 进行测试

在由Slurm管理的集群上，可以使用 `slurm_test.sh` 生成测试工作。它支持单节点和多节点测试。

基本用法如下。

```shell script
bash ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${GPUS}
```

用 Slurm 测试 VIS 模型 MaskTrack R-CNN 的例子：

```shell script
PORT=29501 \
GPUS_PER_NODE=8 \
SRUN_ARGS="--quotatype=reserved" \
bash ./tools/slurm_test.sh \
mypartition \
masktrack \
configs/vis/masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py \
8 \
--checkpoint https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth
```
