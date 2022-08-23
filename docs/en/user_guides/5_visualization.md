# Learn about Visualization

## Local Visualization

This section will present how to visualize the detection/tracking results with local visualizer.

You can turn this feature on by setting `draw=True` in `TrackVisualizationHook` as follows.

```shell script
default_hooks = dict(visualization=dict(type='TrackVisualizationHook', draw=True))
```

Specifically, the `TrackVisualizationHook` has the following arguments:

- `draw`: whether to draw prediction results. If it is False, it means that no drawing will be done. Defaults to False.
- `interval`: The interval of visualization. Defaults to 30.
- `score_thr`: The threshold to visualize the bboxes and masks. Defaults to 0.3.
- `show`: Whether to display the drawn image. Default to False.
- `wait_time`: The interval of show (s). Defaults to 0.
- `test_out_dir`: directory where painted images will be saved in testing process.
- `file_client_args`: Arguments to instantiate a FileClient. Defaults to `dict(backend='disk')`.

In the `TrackVisualizationHook`, a visualizer will be called to implement visualization,
i.e., `DetLocalVisualizer` for VID task and `TrackLocalVisualizer` for MOT, VIS, SOT, VOS tasks.
We will present the details below.

#### Detection Visualization

We realize the detection visualization with class `DetLocalVisualizer`.
You can call it as follows.

```python
visualizer = dict(type='DetLocalVisualizer')
```

It has the following arguments:

- `name`: Name of the instance. Defaults to 'visualizer'.
- `image`: the origin image to draw. The format should be RGB. Defaults to None.
- `vis_backends`: Visual backend config list. Defaults to None.
- `save_dir`: Save file dir for all storage backends. If it is None, the backend storage will not save any data.
- `bbox_color`: Color of bbox lines. The tuple of color should be in BGR order. Defaults to None.
- `text_color`: Color of texts. The tuple of color should be in BGR order. Defaults to (200, 200, 200).
- `line_width`: The linewidth of lines. Defaults to 3.
- `alpha`: The transparency of bboxes or mask. Defaults to 0.8.

Here is a visualization example of DFF:

![test_img_29](https://user-images.githubusercontent.com/99722489/186062793-623f6b1e-163e-4e1a-aa79-efea2d97a16d.png)

#### Tracking Visualization

We realize the tracking visualization with class `TrackLocalVisualizer`.
You can call it as follows.

```python
visualizer = dict(type='TrackLocalVisualizer')
```

It has the following arguments, which has the same meaning of that in `DetLocalVisualizer`.

`name`, `image`, `vis_backends`, `save_dir`, `line_width`, `alpha`.

Here is a visualization example of DeepSORT:

![test_img_89](https://user-images.githubusercontent.com/99722489/186062929-6d0e4663-0d8e-4045-9ec8-67e0e41da876.png)
