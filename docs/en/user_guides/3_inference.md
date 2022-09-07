# Inference

We provide demo scripts to inference a given video or a folder that contains continuous images. The source codes are available [here](https://github.com/open-mmlab/mmtracking/tree/dev-1.x/demo/).

Note that if you use a folder as the input, the image names there must be  **sortable** , which means we can re-order the images according to the numbers contained in the filenames. We now only support reading the images whose filenames end with `.jpg`, `.jpeg` and `.png`.

## Inference VID models

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

The `INPUT` and `OUTPUT` support both _mp4 video_ format and the _folder_ format.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.

**Examples:**

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`, your video filename is `demo.mp4`, and your output path is the `./outputs/`

```shell
python ./demo/demo_vid.py \
    configs/vid/selsa/selsa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py \
    --input ./demo.mp4 \
    --checkpoint checkpoints/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth \
    --output ./outputs/ \
    --show
```

## Inference MOT/VIS models

This script can inference an input video / images with a multiple object tracking or video instance segmentation model.

```shell
python demo/demo_mot_vis.py \
    ${CONFIG_FILE} \
    --input ${INPUT} \
    [--output ${OUTPUT}] \
    [--checkpoint ${CHECKPOINT_FILE}] \
    [--score-thr ${SCORE_THR} \
    [--device ${DEVICE}] \
    [--show]
```

The `INPUT` and `OUTPUT` support both _mp4 video_ format and the _folder_ format.

**Important:** For `DeepSORT`, `SORT`, `Tracktor`, `StrongSORT`, they need both the weight of the `reid` and the weight of the `detector`. Therefore, we can't use `--checkpoint` to specify it. We need to use `init_cfg` in the configuration file to set the weight path. Other algorithms such as `ByteTrack`, `OCSORT` and `QDTrack` need not pay attention to this.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `CHECKPOINT_FILE`: The checkpoint is optional in case that you already set up the pretrained models in the config by the key `init_cfg`.
- `SCORE_THR`: The threshold of score to filter bboxes.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.

**Examples of running mot model:**

```shell
# Example 1: do not specify --checkpoint to use the default init_cfg
python demo/demo_mot_vis.py \
    configs/mot/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4

# Example 2: use --checkpoint
python demo/demo_mot_vis.py \
    configs/mot/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --checkpoint checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth \
    --output mot.mp4
```

**Examples of running vis model:**

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`, your video filename is `demo.mp4`, and your output path is the `./outputs/`

```shell
python demo/demo_mot_vis.py \
    configs/vis/masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py \
    --input demo.mp4 \
    --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
    --output ./outputs/ \
    --show
```

## Inference SOT models

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

The `INPUT` and `OUTPUT` support both _mp4 video_ format and the _folder_ format.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.
- `GT_BBOX_FILE`: The gt_bbox file path of the video. We only use the gt_bbox of the first frame. If not specified, you would draw init bbox of the video manually.

**Examples:**

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`

```shell
python ./demo/demo_sot.py \
    configs/sot/siamese_rpn/siamese-rpn_r50_8xb28-20e_imagenetvid-imagenetdet-coco_test-lasot.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
    --output ${OUTPUT} \
    --show
```
