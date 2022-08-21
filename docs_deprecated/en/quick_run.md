## Run with Existing Datasets and Models

MMTracking provides various methods on existing benchmarks.
Details about these methods and benchmarks are presented in [model_zoo.md](https://mmtracking.readthedocs.io/en/latest/model_zoo.html).
This note will show how to perform common tasks on existing models and standard datasets, including:

- Inference existing models on a given video or image folder.
- Test (inference and evaluate) existing models on standard datasets.
- Train existing models on standard datasets.

### Inference

We provide demo scripts to inference a given video or a folder that contains continuous images.
The source codes are available [here](https://github.com/open-mmlab/mmtracking/tree/master/demo/).

Note that if you use a folder as the input, the image names there must be **sortable**, which means we can re-order the images according to the numbers contained in the filenames. We now only support reading the images whose filenames end with '.jpg', '.jpeg' and '.png'.

#### Inference VID models

This script can inference an input video with a video object detection model.

```
python demo/demo_vid.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --checkpoint ${CHECKPOINT_FILE} \
    [--output ${OUTPUT}] \
    [--device ${DEVICE}] \
    [--show]
```

The `INPUT` and `OUTPUT` support both mp4 video format and the folder format.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`

```shell
python ./demo/demo_vid.py \
    ./configs/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth \
    --output ${OUTPUT} \
    --show
```

#### Inference MOT/VIS models

This script can inference an input video / images with a multiple object tracking or video instance segmentation model.

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

The `INPUT` and `OUTPUT` support both mp4 video format and the folder format.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `CHECKPOINT_FILE`: The checkpoint is optional in case that you already set up the pretrained models in the config by the key `pretrains`.
- `SCORE_THR`: The threshold of score to filter bboxes.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `BACKEND`: The backend to visualize the boxes. Options are `cv2` and `plt`.
- `--show`: Whether show the video on the fly.

Examples of running mot model:

```shell
python demo/demo_mot_vis.py \
    configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py \
    --input demo/demo.mp4 \
    --output mot.mp4 \
```

**Important**: When running `demo_mot_vis.py`, we suggest you use the config containing `private`, since `private` means the MOT method doesn't need external detections.

Examples of running vis model:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`

```shell
python demo/demo_mot_vis.py \
    configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
    --output ${OUTPUT} \
    --show
```

#### Inference SOT models

This script can inference an input video with a single object tracking model.

```shell
python demo/demo_sot.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --checkpoint ${CHECKPOINT_FILE} \
    [--output ${OUTPUT}] \
    [--device ${DEVICE}] \
    [--show] \
    [--gt_bbox_file ${GT_BBOX_FILE}]
```

The `INPUT` and `OUTPUT` support both mp4 video format and the folder format.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.
- `--gt_bbox_file`: The gt_bbox file path of the video. We only use the gt_bbox of the first frame. If not specified, you would draw init bbox of the video manually.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`

```shell
python ./demo/demo_sot.py \
    ./configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
    --output ${OUTPUT} \
    --show
```

### Testing

This section will show how to test existing models on supported datasets.
The following testing environments are supported:

- single GPU
- single node multiple GPU
- multiple nodes

During testing, different tasks share the same API and we only support `samples_per_gpu = 1`.

You can use the following commands for testing:

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Optional arguments:

- `CHECKPOINT_FILE`: Filename of the checkpoint. You do not need to define it when applying some MOT methods but specify the checkpoints in the config.
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `bbox` is available for ImageNet VID, `track` is available for LaSOT, `bbox` and `track` are both suitable for MOT17.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file
- `--eval-options`: If specified, the key-value pair optional eval cfg will be kwargs for dataset.evaluate() function, it’s only for evaluation
- `--format-only`: If specified, the results will be formatted to the official format.

#### Examples of testing VID model

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test DFF on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   python tools/test.py configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```

2. Test DFF with 8 GPUs on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   ./tools/dist_test.sh configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py 8 \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```

#### Examples of testing MOT model

1. Test Tracktor on MOT17, and evaluate CLEAR MOT metrics.

   ```shell
   python tools/test.py configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py \
       --eval track
   ```

2. Test Tracktor with 8 GPUs on MOT17, and evaluate CLEAR MOT metrics.

   ```shell
   ./tools/dist_test.sh configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py 8 \
       --eval track
   ```

3. If you want to test Tracktor with your detector and reid model, you need modify the corresponding key-value pair in config as follows:

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

#### Examples of testing SOT model

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test SiameseRPN++ on LaSOT, and evaluate the success, precision and normed precision.

   ```shell
   python tools/test.py configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py \
       --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
       --out results.pkl \
       --eval track
   ```

2. Test SiameseRPN++ with 8 GPUs on LaSOT, and evaluate the success, precision and normed precision.

   ```shell
   ./tools/dist_test.sh configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py 8 \
       --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
       --out results.pkl \
       --eval track
   ```

#### Examples of testing VIS model

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test MaskTrack R-CNN on YouTube-VIS 2019, and generate a zip file for submission.

   ```shell
   python tools/test.py \
       configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py \
       --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
       --out ${RESULTS_PATH}/results.pkl \
       --format-only \
       --eval-options resfile_path=${RESULTS_PATH}
   ```

2. Test MaskTrack R-CNN with 8 GPUs on YouTube-VIS 2019, and generate a zip file for submission.

   ```shell
   ./tools/dist_test.sh \
       configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py \
       --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
       --out ${RESULTS_PATH}/results.pkl \
       --format-only \
       --eval-options resfile_path=${RESULTS_PATH}
   ```

### Training

MMTracking also provides out-of-the-box tools for training models.
This section will show how to train _predefined_ models (under [configs](https://github.com/open-mmlab/mmtracking/tree/master/configs)) on standard datasets.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.

```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

**Important**: The default learning rate in all config files is for 8 GPUs.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., `lr=0.01` for 8 GPUs * 1 img/gpu and `lr=0.04` for 16 GPUs * 2 imgs/gpu.

#### Training on a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file or via CLI argument `--work-dir`.

#### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

Optional arguments remain the same as stated above.

If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

#### Training on multiple nodes

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

Usually it is slow if you do not have high speed networking like InfiniBand.

If you launch with slurm, the command is the same as that on single machine described above, but you need refer to [slurm_train.sh](https://github.com/open-mmlab/mmtracking/blob/master/tools/slurm_train.sh) to set appropriate parameters and environment variables.

#### Manage jobs with Slurm

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.

The basic usage is as follows.

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

You can check [the source code](https://github.com/open-mmlab/mmtracking/blob/master/tools/slurm_train.sh) to review full arguments and environment variables.

When using Slurm, the port option need to be set in one of the following ways:

1. Set the port through `--options`. This is more recommended since it does not change the original configs.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
   ```

2. Modify the config files to set different communication ports.

   In `config1.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   In `config2.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   Then you can launch two jobs with `config1.py` and `config2.py`.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

#### Examples of training VID model

1. Train DFF on ImageNet VID and ImageNet DET, then evaluate the bbox mAP at the last epoch.

   ```shell
   bash ./tools/dist_train.sh ./configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py 8 \
       --work-dir ./work_dirs/
   ```

#### Examples of training MOT model

For the training of MOT methods like SORT, DeepSORT and Tracktor, you need train a detector and a reid model rather than directly training the MOT model itself.

1. Train a detector model

   If you want to train a detector for multiple object tracking or other applications, to be compatible with MMDetection, you only need to add a line of `USE_MMDET=True` in the config and run it with the same manner in mmdetection.
   A base example can be found at [faster_rcnn_r50_fpn.py](https://github.com/open-mmlab/mmtracking/blob/master/configs/_base_/models/faster_rcnn_r50_fpn.py).

   Please NOTE that there are some differences between the base config in MMTracking and MMDetection: `detector` is only a submodule of the `model`.
   For example, the config of Faster R-CNN in MMDetection follows

   ```python
   model = dict(
       type='FasterRCNN',
       ...
   )
   ```

   But in MMTracking, the config follows

   ```python
   model = dict(
       detector=dict(
           type='FasterRCNN',
           ...
       )
   )
   ```

   Here is an example to train a detector model on MOT17, and evaluate the bbox mAP after each epoch.

   ```shell
   bash ./tools/dist_train.sh ./configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py 8 \
       --work-dir ./work_dirs/
   ```

2. Train a ReID model

   You may want to train a ReID model for multiple object tracking or other applications. We support ReID model training in MMTracking, which is built upon [MMClassification](https://github.com/open-mmlab/mmclassification).

   Here is an example to train a reid model on MOT17, then evaluate the mAP after each epoch.

   ```shell
   bash ./tools/dist_train.sh ./configs/reid/resnet50_b32x8_MOT17.py 8 \
       --work-dir ./work_dirs/
   ```

3. After training a detector and a ReID model, you can refer to [Examples of testing MOT model](https://mmtracking.readthedocs.io/en/latest/quick_run.html#examples-of-testing-mot-model) to test your multi-object tracker.

#### Examples of training SOT model

1. Train SiameseRPN++ on COCO, ImageNet VID and ImageNet DET, then evaluate the success, precision and normed precision from the 10-th epoch to 20-th epoch.

   ```shell
   bash ./tools/dist_train.sh ./configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py 8 \
       --work-dir ./work_dirs/
   ```

#### Examples of training VIS model

1. Train MaskTrack R-CNN on YouTube-VIS 2019 dataset. There are no evaluation results during training, since the annotations of validation dataset in YouTube-VIS are not provided.

   ```shell
   bash ./tools/dist_train.sh ./configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py 8 \
       --work-dir ./work_dirs/
   ```

## Run with Customized Datasets and Models

In this note, you will know how to inference, test, and train with customized datasets and models.

The basic steps are as below:

1. Prepare the customized dataset (if applicable)
2. Prepare the customized model (if applicable)
3. Prepare a config
4. Train a new model
5. Test and inference the new model

### 1. Prepare the customized dataset

There are two ways to support a new dataset in MMTracking:

Reorganize the dataset into CocoVID format.
Implement a new dataset.

Usually we recommend to use the first method which is usually easier than the second.

Details for customizing datasets are provided in [tutorials/customize_dataset.md](https://mmtracking.readthedocs.io/en/latest/tutorials/customize_dataset.html).

### 2. Prepare the customized model

We provide instructions for cutomizing models of different tasks.

- [tutorials/customize_vid_model.md](https://mmtracking.readthedocs.io/en/latest/tutorials/customize_vid_model.html)
- [tutorials/customize_mot_model.md](https://mmtracking.readthedocs.io/en/latest/tutorials/customize_mot_model.html)
- [tutorials/customize_sot_model.md](https://mmtracking.readthedocs.io/en/latest/tutorials/customize_sot_model.html)

### 3. Prepare a config

The next step is to prepare a config thus the dataset or the model can be successfully loaded.
More details about the config system are provided at [tutorials/config.md](https://mmtracking.readthedocs.io/en/latest/tutorials/config.html).

### 4. Train a new model

To train a model with the new config, you can simply run

```shell
python tools/train.py ${NEW_CONFIG_FILE}
```

For more detailed usages, please refer to the training instructions above.

### 5. Test and inference the new model

To test the trained model, you can simply run

```shell
python tools/test.py ${NEW_CONFIG_FILE} ${TRAINED_MODEL} --eval bbox track
```

For more detailed usages, please refer to the testing or inference instructions above.
