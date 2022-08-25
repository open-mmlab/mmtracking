在`tools/`目录下我们提供了许多有用的工具。

## MOT 测试时参数搜索

`tools/analysis/mot/mot_param_search.py` 脚本可以搜索 MOT 模型中`跟踪器`的参数。
除了在配置文件上有所差别，该脚本的使用方法与 `tools/test.py` 脚本类似。

这里有一个示例来展示如何修改配置文件：

1. 定义所需的评测指标。

   比如你可以定义如下评测指标

   ```python
   search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
   ```

2. 定义需要被搜索的参数和数值

   假设你的跟踪器如下所示

   ```python
   model = dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=0.5,
           match_iou_thr=0.5
       )
   )
   ```

   如果你想搜索这个跟踪器的参数，只需要将对应数值改成列表即可

   ```python
   model = dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=[0.4, 0.5, 0.6],
           match_iou_thr=[0.4, 0.5, 0.6, 0.7]
       )
   )
   ```

   脚本将测试共12个案例，并记录相应测试结果。

## SiameseRPN++ 测试时参数搜索

`tools/analysis/sot/sot_siamrpn_param_search.py` 用来搜索 SiameseRPN++ 测试时的跟踪相关参数： `penalty_k`, `lr` 和 `window_influence`。你需要在参数解析器中传入前面每个参数的搜索范围。

在 UAV123 上的超参搜索范例：

```shell
./tools/analysis/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.22,0.05] [--lr-range 0.4,0.61,0.05] [--win-infu-range 0.01,0.22,0.05]
```

在 OTB100 上的超参搜索范例：

```shell
./tools/analysis/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.3,0.45,0.02] [--lr-range 0.35,0.5,0.02] [--win-infu-range 0.46,0.55,0.02]
```

在 VOT2018 上的超参搜索范例：

```shell
./tools/analysis/sot/dist_sot_siamrpn_param_search.sh [${CONFIG_FILE}] [$GPUS] \
[--checkpoint ${CHECKPOINT}] [--log ${LOG_FILENAME}] [--eval ${EVAL}] \
[--penalty-k-range 0.01,0.31,0.05] [--lr-range 0.2,0.51,0.05] [--win-infu-range 0.3,0.56,0.05]
```

## 日志分析

`tools/analysis/analyze_logs.py` 脚本可以根据训练日志文件绘制损失函数以及 mAP 曲线。

```shell
python tools/analysis/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

几个例子：

- 绘制某次运行时的分类损失函数。

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- 绘制某次运行时的分类以及回归损失函数，并且保存成 pdf 文件。

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- 在同一张图中比较两次运行的 bbox mAP。

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
  ```

- 计算平均运行速度

  ```shell
  python tools/analysis/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  输出如下所示：

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```

## 模型转换

### 发布模型

`tools/analysis/publish_model.py` 脚本可以帮助用户发布模型。

在将模型上传到AWS之前，你可能想要做以下事情：

1. 将模型参数转化成 CPU 张量
2. 删除优化器状态参数
3. 计算模型权重文件的哈希值并将其添加进文件名。

```shell
python tools/analysis/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

比如：

```shell
python tools/analysis/publish_model.py work_dirs/dff_faster_rcnn_r101_dc5_1x_imagenetvid/latest.pth dff_faster_rcnn_r101_dc5_1x_imagenetvid.pth
```

最后输出的文件名为 `dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201230-{hash id}.pth`。

## 其它有用的工具脚本

### 输出完整的配置

`tools/analysis/print_config.py` 脚本可以输出完整的配置，包括文件中导入的配置。

```shell
python tools/analysis/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
