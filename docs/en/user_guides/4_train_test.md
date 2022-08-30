# Learn to train and test

## Train

This section will show how to train existing models on supported datasets.
The following training environments are supported:

- CPU
- single GPU
- single node multiple GPUs
- multiple nodes

You can also manage jobs with Slurm.

Important:

- You can change the evaluation interval during training by modifying the `train_cfg` as
  `train_cfg = dict(val_interval=10)`. That means evaluating the model every 10 epochs.
- The default learning rate in all config files is for 8 GPUs.
  According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677),
  you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU,
  e.g., `lr=0.01` for 8 GPUs * 1 img/gpu and lr=0.04 for 16 GPUs * 2 imgs/gpu.
- During training, log files and checkpoints will be saved to the working directory,
  which is specified by CLI argument `--work-dir`. It uses `./work_dirs/CONFIG_NAME` as default.
- If you want the mixed precision training, simply specify CLI argument `--amp`.

#### 1. Train on CPU

The model is default put on cuda device.
Only if there are no cuda devices, the model will be put on cpu.
So if you want to train the model on CPU, you need to `export CUDA_VISIBLE_DEVICES=-1` to disable GPU visibility first.
More details in [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850).

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [optional arguments]
```

An example of training the VID model DFF on CPU:

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/vid/dff/dff_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py
```

#### 2. Train on single GPU

If you want to train the model on single GPU, you can directly use the `tools/train.py` as follows.

```shell script
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

You can use `export CUDA_VISIBLE_DEVICES=$GPU_ID` to select the GPU.

An example of training the MOT model ByteTrack on single GPU:

```shell script
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py
```

#### 3. Train on single node multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell script
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

If you would like to launch multiple jobs on a single machine,
e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

For example, you can set the port in commands as follows.

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

An example of training the SOT model SiameseRPN++ on single node multiple GPUs:

```shell script
bash ./tools/dist_train.sh ./configs/sot/siamese_rpn/siamese-rpn_r50_8xb16-20e_imagenetvid-imagenetdet-coco_test-otb100.py 8
```

#### 4. Train on multiple nodes

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell script
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell script
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

Usually it is slow if you do not have high speed networking like InfiniBand.

#### 5. Train with Slurm

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs.
It supports both single-node and multi-node training.

The basic usage is as follows.

```shell script
bash ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPUS}
```

An example of training the VIS model MaskTrack R-CNN with Slurm:

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

## Test

This section will show how to test existing models on supported datasets.
The following testing environments are supported:

- CPU
- single GPU
- single node multiple GPUs
- multiple nodes

You can also manage jobs with Slurm.

Important:

- You can set the results saving path by modifying the key `outfile_prefix` in evaluator.
  For example, `val_evaluator = dict(outfile_prefix='results/stark_st1_trackingnet')`.
  Otherwise, a temporal file will be created and will be removed after evaluation.
- If you just want the formatted results without evaluation, you can set `format_only=True`.
  For example, `test_evaluator = dict(type='YouTubeVISMetric', metric='youtube_vis_ap', outfile_prefix='./youtube_vis_results', format_only=True)`

#### 1. Test on CPU

The model is default put on cuda device.
Only if there are no cuda devices, the model will be put on cpu.
So if you want to test the model on CPU, you need to `export CUDA_VISIBLE_DEVICES=-1` to disable GPU visibility first.
More details in [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/runner/runner.py#L849-L850).

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} [optional arguments]
```

An example of testing the VID model DFF on CPU:

```shell script
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/vid/dff/dff_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py --checkpoint https://download.openmmlab.com/mmtracking/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid/dff_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_213250-548911a4.pth
```

#### 2. Test on single GPU

If you want to test the model on single GPU, you can directly use the `tools/test.py` as follows.

```shell script
python tools/test.py ${CONFIG_FILE} [optional arguments]
```

You can use `export CUDA_VISIBLE_DEVICES=$GPU_ID` to select the GPU.

An example of testing the MOT model ByteTrack on single GPU:

```shell script
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py --checkpoint https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
```

#### 3. Test on single node multiple GPUs

We provide `tools/dist_test.sh` to launch testing on multiple GPUs.
The basic usage is as follows.

```shell script
bash ./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

An example of testing the SOT model SiameseRPN++ on single node multiple GPUs:

```shell script
bash ./tools/dist_test.sh ./configs/sot/siamese_rpn/siamese-rpn_r50_8xb16-20e_imagenetvid-imagenetdet-coco_test-otb100.py 8 --checkpoint https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_20e_otb100_20220421_144232-6b8f1730.pth
```

#### 4. Test on multiple nodes

You can test on multiple nodes, which is similar with "Train on multiple nodes".

#### 5. Test with Slurm

On a cluster managed by Slurm, you can use `slurm_test.sh` to spawn testing jobs.
It supports both single-node and multi-node testing.

The basic usage is as follows.

```shell script
bash ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${GPUS}
```

An example of testing the VIS model MaskTrack R-CNN with Slurm:

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
