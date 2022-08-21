We provide lots of useful tools under the `tools/` directory.

## MOT Test-time Parameter Search

`tools/analysis/mot/mot_param_search.py` can search the parameters of the `tracker` in MOT models.
It is used as the same manner with `tools/test.py` but different in the configs.

Here is an example that shows how to modify the configs:

1. Define the desirable evaluation metrics to record.

   For example, you can define the search metrics as

   ```python
   search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
   ```

2. Define the parameters and the values to search.

   Assume you have a tracker like

   ```python
   model = dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=0.5,
           match_iou_thr=0.5
       )
   )
   ```

   If you want to search the parameters of the tracker, just change the value to a list as follow

   ```python
   model = dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=[0.4, 0.5, 0.6],
           match_iou_thr=[0.4, 0.5, 0.6, 0.7]
       )
   )
   ```

   Then the script will test the totally 12 cases and log the results.

## SiameseRPN++ Test-time Parameter Search

`tools/analysis/sot/sot_siamrpn_param_search.py` can search the test-time tracking parameters in SiameseRPN++: `penalty_k`, `lr` and `window_influence`. You need to pass the searching range of each parameter into the argparser.

Example on UAV123 dataset:

```shell
./tools/analysis/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.22,0.05] [--lr-range 0.4,0.61,0.05] [--win-infu-range 0.01,0.22,0.05]
```

Example on OTB100 dataset:

```shell
./tools/analysis/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.3,0.45,0.02] [--lr-range 0.35,0.5,0.02] [--win-infu-range 0.46,0.55,0.02]
```

Example on VOT2018 dataset:

```shell
./tools/analysis/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.31,0.05] [--lr-range 0.2,0.51,0.05] [--win-infu-range 0.3,0.56,0.05]
```

## Log Analysis

`tools/analysis/analyze_logs.py` plots loss/mAP curves given a training log file.

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- Compare the bbox mAP of two runs in the same figure.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
  ```

- Compute the average training speed.

  ```shell
  python tools/analysis/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  The output is expected to be like the following.

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```

## Model Conversion

### Prepare a model for publishing

`tools/analysis/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/analysis/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/analysis/publish_model.py work_dirs/dff_faster_rcnn_r101_dc5_1x_imagenetvid/latest.pth dff_faster_rcnn_r101_dc5_1x_imagenetvid.pth
```

The final output filename will be `dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201230-{hash id}.pth`.

## Miscellaneous

### Print the entire config

`tools/analysis/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/analysis/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

## Model Serving

In order to serve an `MMTracking` model with [`TorchServe`](https://pytorch.org/serve/), you can follow the steps:

### 1. Convert model from MMTracking to TorchServe

```shell
python tools/torchserve/mmtrack2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

${MODEL_STORE} needs to be an absolute path to a folder.

### 2. Build `mmtrack-serve` docker image

```shell
docker build -t mmtrack-serve:latest docker/serve/
```

### 3. Run `mmtrack-serve`

Check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

In order to run in GPU, you need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). You can omit the `--gpus` argument in order to run in CPU.

Example:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmtrack-serve:latest
```

[Read the docs](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md) about the Inference (8080), Management (8081) and Metrics (8082) APIs

### 4. Test deployment

```shell
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T demo/demo.mp4 -o result.mp4
```

The response will be a ".mp4" mask.

You can visualize the output as follows:

```python
import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        cv2.imshow('result.mp4', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

And you can use `test_torchserve.py` to compare result of torchserve and pytorch, and visualize them.

```shell
python tools/torchserve/test_torchserve.py ${VIDEO_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--result-video ${RESULT_VIDEO}] [--device ${DEVICE}]
[--score-thr ${SCORE_THR}]
```

Example:

```shell
python tools/torchserve/test_torchserve.py \
demo/demo.mp4 \
configs/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py \
checkpoint/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth \
selsa \
--result-video=result.mp4
```
