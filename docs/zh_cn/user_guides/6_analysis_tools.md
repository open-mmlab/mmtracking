**我们在 `tools/` 目录下提供了很多有用的工具。**

## 多目标跟踪测试时的参数搜索

`tools/analysis_tools/mot/mot_param_search.py` 能够搜索 MOT 模型中 `tracker` 参数。
它的使用方式与 `tools/test.py` 一样，但在配置中**有所不同**。

以下是一个展示如何修改配置的示例：

1. 定义所需的评估指标。

   例如，您可以定义 `evaluator` 为

   ```python
   test_evaluator=dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
   ```

   当然，您也可以在 `test_evaluator` 中自定义 `metric` 的内容。您可以自由选择 `['HOTA', 'CLEAR', 'Identity']` 的一个或多个。

2. 定义要搜索的参数和值。

   假定您有一个 tracker 如下：

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=0.5,
           match_iou_thr=0.5
       )
   )
   ```

   如果您想搜索该 tracker 的参数，只需把参数值改成一个列表如下所示：

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=[0.4, 0.5, 0.6],
           match_iou_thr=[0.4, 0.5, 0.6, 0.7]
       )
   )
   ```

   然后，脚本将测试总共 12 个案例，并记录结果。

## 多目标跟踪错误可视化

`tools/analysis_tools/mot/mot_error_visualize.py` 能可视化多目标跟踪的误差。
这个脚本需要已保存的推理结果。默认情况下，**红色**边界框表示假阳性，**黄色**边界框表示假阴性，**蓝色**边界框表示 ID 切换。

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

`RESULT_DIR` 包含所有视频的推理结果，推理结果是一个 `txt` 文件。

可选参数：

- `OUTPUT`: 可视化演示的输出。 如果未指定，`--show` 会强制实时显示视频。
- `FPS`: 输出视频的FPS。
- `--show`: 是否实时显示视频。
- `BACKEND`: 用于可视化框的后端。选项是 `cv2` 和 `plt`。

## SiameseRPN++ 测试时的参数搜索

`tools/analysis_tools/sot/sot_siamrpn_param_search.py` 能够在SiameseRPN++中搜索测试时的跟踪参数: `penalty_k`, `lr` 和 `window_influence`。 您需要将每个参数的搜索范围传递给参数器。

**UAV123 数据集的例子：**

```shell
./tools/analysis_tools/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.22,0.05] [--lr-range 0.4,0.61,0.05] [--win-infu-range 0.01,0.22,0.05]
```

**OTB100 数据集的例子：**

```shell
./tools/analysis_tools/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.3,0.45,0.02] [--lr-range 0.35,0.5,0.02] [--win-infu-range 0.46,0.55,0.02]
```

**VOT2018 数据集的例子：**

```shell
./tools/analysis_tools/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.31,0.05] [--lr-range 0.2,0.51,0.05] [--win-infu-range 0.3,0.56,0.05]
```

## 日志分析

`tools/analysis_tools/analyze_logs.py` 绘制给定训练日志文件的 loss/mAP 曲线。

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

**例子：**

- 绘制某次运行的分类损失。

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- 绘制某次运行的分类和回归损失并将图片保存为 pdf 文件。

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- 在一张图上比较两次运行的 bbox 的 mAP。

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
  ```

- 计算平均训练速度。

  ```shell
  python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  其输出如下所示：

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```

## 浏览数据集

`tools/analysis_tools/browse_dataset.py` 能可视化训练数据集来检查数据及配置是否正确。

**举例：**

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG_FILE} [--show-interval ${SHOW_INTERVAL}]
```

可选参数：

- `SHOW_INTERVAL`: 显示间隔（秒）。
- `--not-show`: 不实时显示图片。

## 在视频等级上展示单目标跟踪评估结果。

单目标跟踪评估结果在视频级别上按成功指标从最大到最小排序。
通过设置 `eval_show_video_indices`，可以有选择地显示一些好情况或坏情况的性能结果。

```python
test_evaluator=dict(
    type='SOTMetric',
    options_after_eval=dict(eval_show_video_indices=10))
```

这里，`eval_show_video_indices` 用于索引 `numpy.ndarray`。
它可以是 `int` (正或负)。正数 `k` 表示最好的 k 个结果，而负数表示最差的 k 个结果。

## 保存并绘制单目标跟踪的评估结果

通过设置配置文件中的 `SOTMetric` 来保存单目标跟踪的评估结果。

```python
test_evaluator = dict(
    type='SOTMetric',
    options_after_eval = dict(tracker_name = 'SiamRPN++', saved_eval_res_file = './results/sot_results.json'))
```

保存的结果是一个字典的格式:

```python
dict{tracker_name=dict(
      success = np.ndarray,
      norm_precision = np.ndarray,
      precision = np.ndarray)}
```

指标形状为（M，），M是不同阈值对应结果的数量。

您可以使用以下命令绘制给定的保存结果：

```shell
python ./tools/analysis_tools/sot/sot_plot_curve.py ./results --plot_save_path ./results
```

# 保存并重现跟踪结果

通过设置配置文件中的 `SOTMetric` 保存跟踪结果。

```python
test_evaluator = dict(
    type='SOTMetric',
    options_after_eval = dict(saved_track_res_path='./tracked_results'))
```

使用以下命令重现跟踪结果：

```shell
python ./tools/analysis_tools/sot/sot_playback.py  data/OTB100/data/Basketball/img/ tracked_results/basketball.txt --show --output results/basketball.mp4 --fps 20 --gt_bboxes data/OTB100/data/Basketball/groundtruth_rect.txt
```

## 特征图的可视化

这是在 mmengine 中调用可视化工具的示例：

```python
# 在任何位置调用可视化器
visualizer = Visualizer.get_current_instance()
# 设置图像为背景
visualizer.set_image(image=image)
# 在图像上绘制特征图
drawn_img = visualizer.draw_featmap(feature_map, image, channel_reduction='squeeze_mean')
# 显示
visualizer.show(drawn_img)
# 存储为 ${saved_dir}/vis_data/vis_image/feat_0.png
visualizer.add_image('feature_map', drawn_img)
```

有关特征图可视化的更多详细信息，请参见 [visualizer 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md) 和 [draw_featmap 函数](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/visualizer.py#L864)
