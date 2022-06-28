## 自定义视频目标检测模型

我们通常将模型组件分为3类：

- 检测器：通常是从一张图片中检出物体的检测组件，例如：Faster R-CNN。
- 运动估计器：计算两张图片之间的运动信息的组件，例如：FlowNetSimple。
- 聚合器：聚合多张图片特征的组件，例如：EmbedAggregator。

### 增加一个新的检测器

请参考[MMDetection教程](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/customize_models.html)来开发新检测器

### 增加一个新的运动估计器

#### 1. 定义一个运动估计模型（例如：MyFlowNet）

新建一个文件 `mmtrack/models/motion/my_flownet.py`。

```python
from mmcv.runner import BaseModule

from ..builder import MOTION

@MOTION.register_module()
class MyFlowNet(BaseModule):

    def __init__(self,
                arg1,
                arg2):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引入模块

你可以在 `mmtrack/models/motion/__init__.py` 中加入下面一行。

```python
from .my_flownet import MyFlowNet
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.motion.my_flownet.py'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config 文件

```python
motion=dict(
    type='MyFlowNet',
    arg1=xxx,
    arg2=xxx)
```

### 增加一个新的聚合器

#### 1. 定义一个聚合器

创建一个新文件 `mmtrack/models/aggregators/my_aggregator.py`。

```python
from mmcv.runner import BaseModule

from ..builder import AGGREGATORS

@AGGREGATORS.register_module()
class MyAggregator(BaseModule):

    def __init__(self,
                arg1,
                arg2):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引入模块

你可以在 `mmtrack/models/aggregators/__init__.py` 中加入下面一行。

```python
from .my_aggregator import MyAggregator
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.aggregators.my_aggregator.py'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config 文件

```python
aggregator=dict(
    type='MyAggregator',
    arg1=xxx,
    arg2=xxx)
```
