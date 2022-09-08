**We provide lots of useful tools under the `tools/` directory.**

## MOT Test-time Parameter Search

`tools/analysis_tools/mot/mot_param_search.py` can search the parameters of the `tracker` in MOT models.
It is used as the same manner with `tools/test.py` but **different** in the configs.

Here is an example that shows how to modify the configs:

1. Define the desirable evaluation metrics to record.

   For example, you can define the `evaluator` as

   ```python
   test_evaluator=dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
   ```

   Of course, you can also customize the content of `metric` in `test_evaluator`. You are free to choose one or more of `['HOTA', 'CLEAR', 'Identity']`.

2. Define the parameters and the values to search.

   Assume you have a tracker like

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=0.5,
           match_iou_thr=0.5
       )
   )
   ```

   If you want to search the parameters of the tracker, just change the value to a list as follow

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=[0.4, 0.5, 0.6],
           match_iou_thr=[0.4, 0.5, 0.6, 0.7]
       )
   )
   ```

   Then the script will test the totally 12 cases and log the results.

## MOT Error Visualize

`tools/analysis_tools/mot/mot_error_visualize.py` can visualize errors for multiple object tracking.
This script needs the result of inference. By Default, the **red** bounding box denotes false positive, the **yellow** bounding box denotes the false negative and the **blue** bounding box denotes ID switch.

```
python tools/analysis_tools/mot/mot_error_visualize.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --result-dir ${RESULT_DIR} \
    [--out-dir ${OUTPUT}] \
    [--fps ${FPS}] \
    [--show] \
    [--backend ${BACKEND}]
```

The `RESULT_DIR` contains the inference results of all videos and the inference result is a `txt` file.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `FPS`: FPS of the output video.
- `--show`: Whether show the video on the fly.
- `BACKEND`: The backend to visualize the boxes. Options are `cv2` and `plt`.

## SiameseRPN++ Test-time Parameter Search

`tools/analysis_tools/sot/sot_siamrpn_param_search.py` can search the test-time tracking parameters in SiameseRPN++: `penalty_k`, `lr` and `window_influence`. You need to pass the searching range of each parameter into the argparser.

**Example on UAV123 dataset:**

```shell
./tools/analysis_tools/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.22,0.05] [--lr-range 0.4,0.61,0.05] [--win-infu-range 0.01,0.22,0.05]
```

**Example on OTB100 dataset:**

```shell
./tools/analysis_tools/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.3,0.45,0.02] [--lr-range 0.35,0.5,0.02] [--win-infu-range 0.46,0.55,0.02]
```

**Example on VOT2018 dataset:**

```shell
./tools/analysis_tools/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.31,0.05] [--lr-range 0.2,0.51,0.05] [--win-infu-range 0.3,0.56,0.05]
```

## Log Analysis

`tools/analysis_tools/analyze_logs.py` plots loss/mAP curves given a training log file.

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

**Examples:**

- Plot the classification loss of some run.

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- Compare the bbox mAP of two runs in the same figure.

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
  ```

- Compute the average training speed.

  ```shell
  python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  The output is expected to be like the following.

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```

## Browse dataset

`tools/analysis_tools/browse_dataset.py` can visualize the training dataset to check whether the dataset configuration is correct.

**Examples:**

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG_FILE} [--show-interval ${SHOW_INTERVAL}]
```

Optional arguments:

- `SHOW_INTERVAL`: The interval of show (s).
- `--not-show`: Whether do not show the images on the fly.

## Show SOT evaluation results in video level

The SOT evaluation results are sorted in video level from largest to smallest by the Success metric.
You can selectively show the performance results of some good cases or bad cases by setting `eval_show_video_indices`.

```python
test_evaluator=dict(
    type='SOTMetric',
    options_after_eval=dict(eval_show_video_indices=10))
```

Here, `eval_show_video_indices` is used to index a `numpy.ndarray`.
It can be `int` (positive or negative) or `list`. The positive number `k` means all the top-k
reuslts while the negative number means the bottom-k results.

## Save SOT evaluation results and plot them

Save the SOT evaluation result by setting the `SOTMetric` in the config.

```python
test_evaluator = dict(
    type='SOTMetric',
    options_after_eval = dict(tracker_name = 'SiamRPN++', saved_eval_res_file = './results/sot_results.json'))
```

The saved result is a dict in the format:

```python
dict{tracker_name=dict(
      success = np.ndarray,
      norm_precision = np.ndarray,
      precision = np.ndarray)}
```

The metrics have shape (M, ), where M is the number of values corresponding to different thresholds.

Given the saved results, you can plot them using the following command:

```shell
python ./tools/analysis_tools/sot/sot_plot_curve.py ./results --plot_save_path ./results
```

# Save tracked results and playback them

Save the tracked result by setting the `SOTMetric` in the config.

```python
test_evaluator = dict(
    type='SOTMetric',
    options_after_eval = dict(saved_track_res_path='./tracked_results'))
```

Playback the tracked results using the following command:

```shell
python ./tools/analysis_tools/sot/sot_playback.py  data/OTB100/data/Basketball/img/ tracked_results/basketball.txt --show --output results/basketball.mp4 --fps 20 --gt_bboxes data/OTB100/data/Basketball/groundtruth_rect.txt
```

## Visualization of feature map

Here is an example of calling the Visualizer in MMEngine:

```python
# call visualizer at any position
visualizer = Visualizer.get_current_instance()
# set the image as background
visualizer.set_image(image=image)
# draw feature map on the image
drawn_img = visualizer.draw_featmap(feature_map, image, channel_reduction='squeeze_mean')
# show
visualizer.show(drawn_img)
# saved as ${saved_dir}/vis_data/vis_image/feat_0.png
visualizer.add_image('feature_map', drawn_img)
```

More details about visualization of feature map can be seen in [visualizer docs](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md) and [draw_featmap function](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/visualizer.py#L864)
