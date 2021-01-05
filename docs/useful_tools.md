We provide lots of useful tools under the `tools/` directory.

## MOT Test-time Parameter Search

`tools/mot_param_search.py` can search the parameters of the `tracker` in MOT models.
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

## Log Analysis

`tools/analyze_logs.py` plots loss/mAP curves given a training log file.

 ```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

    ```shell
    python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

    ```shell
    python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
    ```

- Compare the bbox mAP of two runs in the same figure.

    ```shell
    python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
    ```

- Compute the average training speed.

    ```shell
    python tools/analyze_logs.py cal_train_time log.json [--include-outliers]
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

`tools/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append the hash id to the
 filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/dff_faster_rcnn_r101_dc5_1x_imagenetvid/latest.pth dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201230.pth
```

The final output filename will be `dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201230-{hash id}.pth`.

## Miscellaneous

### Print the entire config

`tools/print_config.py` prints the whole config verbatim, expanding all its
 imports.

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
