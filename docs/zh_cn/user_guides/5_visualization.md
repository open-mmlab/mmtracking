# 学习如何进行可视化操作

## 本地可视化

本节将介绍如何使用本地可视化工具可视化检测/跟踪结果。

如果你想绘制预测结果，你可以通过在 `TrackVisualizationHook` 中设置 `draw=True` 打开这个功能，如下所示。

```shell script
default_hooks = dict(visualization=dict(type='TrackVisualizationHook', draw=True))
```

具体来说，`TrackVisualizationHook` 有以下参数：

- `draw`：是否绘制预测结果。如果为False，则表示不进行绘图。默认为False。
- `interval`：可视化的间隔。默认值为30。
- `score_thr`：可视化框和掩码的阈值。默认值为0.3。
- `show`：是否显示绘制的图像。默认为False。
- `wait_time`：展示的间隔。默认值为0。
- `test_out_dir`：在测试过程中绘制的图像将被保存的目录。
- `file_client_args`：用于实例化 FileClient 的参数。默认为 `dict(backend='disk')` 。

在 `TrackVisualizationHook` 中，将调用一个可视化器来实现可视化，

例如，`DetLocalVisualizer` 用于VID任务，`TrackLocalVisualizer` 用于MOT, VIS, SOT, VOS任务。

我们将在下面详细介绍。

关于 [Visualization](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) 和 [Hook](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md) 的更多细节，可以参考MMEngine。

#### 可视化检测

我们用 `DetLocalVisualizer` 来实现检测可视化。

您可以通过这样增加 configs 配置来调用它

```python
visualizer = dict(type='DetLocalVisualizer')
```

它有以下参数：

- `name`：实例的名称。默认为 'visualizer'
- `image`：要绘制的原始图像。格式应该是 RGB 。默认为 None。
- `vis_backends`：可视化后端配置列表。默认为 None。
- `save_dir`：为所有存储后端保存文件 dir 。如果为 None，则后端存储将不保存任何数据。
- `bbox_color`：方框线的颜色。color的元组应该按照 BGR 顺序。默认为 None。
- `text_color`：文字的颜色。color的元组应该按照 BGR 顺序。默认为 (200,200,200)。
- `line_width`：线的宽度。默认值为3。
- `alpha`：掩膜的透明度。默认值为0.8。

下面是一个 DFF 的可视化示例：

![test_img_29](https://user-images.githubusercontent.com/99722489/186062793-623f6b1e-163e-4e1a-aa79-efea2d97a16d.png)

#### 跟踪可视化

我们通过 `TrackLocalVisualizer` 来实现跟踪可视化。

您可以通过这样增加 configs 配置来调用它

```python
visualizer = dict(type='TrackLocalVisualizer')
```

它有以下参数，它们与 `DetLocalVisualizer` 中的含义相同。

`name`, `image`, `vis_backends`, `save_dir`, `line_width`, `alpha`.

下面是一个 DeepSORT 的可视化示例：

![test_img_89](https://user-images.githubusercontent.com/99722489/186062929-6d0e4663-0d8e-4045-9ec8-67e0e41da876.png)
