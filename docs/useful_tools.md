We provide lots of useful tools under `tools/` directory.

## MOT Test-time Parameter Search

## Log Analysis

`tools/analyze_logs.py` plots loss/mAP curves given a training log file.

 ```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

![loss curve image](../resources/loss_curve.png)

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
