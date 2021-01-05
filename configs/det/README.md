# Compatibility with MMDetection

You may want to train a detector for multiple object tracking or other applications.

To be compatablie with MMDetection, you only need to add a line of `USE_MMDET=True` in the config and run it with the same manner in mmdetection.

A base example can be found at [faster_rcnn_r50_fpn.py](../_base_/models/faster_rcnn_r50_fpn.py).

Please NOTE that there are some differences between the base config in MMTracking and MMDetection.

1. `detector` is only a submodule of the `model`.

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

2. `train_cfg` and `test_cfg` are merged into `model` / `detector`.

    In MMDetection, the configs follows

    ```python
    model = dict()
    train_cfg = dict()
    test_cfg = dict()
    ```

    While in MMTracking, the config follows

    ```python
    model = dict(
        detector=dict(
            train_cfg=dict(),
            test_cfg=dict(),
            ...
        )
    )
    ```

We are putting efforts to bridge these gaps across different codebases and try to give simpler configs.
