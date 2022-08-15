## 自定义多目标跟踪模型

我们通常将模型组件分为5类：

- 跟踪器：利用以下组件提取出来的线索来关联视频帧间目标的组件。
- 检测器：通常是从一张图片中检出物体的检测器，例如：Faster R-CNN。
- 运动估计器：计算相邻帧运动信息的组件，例如：卡尔曼滤波器。
- 重识别器：从裁剪图片中抽取特征的的独立重识别模型，例如：BaseReID。
- 跟踪头：用于抽取跟踪线索但是和检测器共享骨干网络的组件。例如：一个特征分支头或者回归分支头。

### 增加一个新的跟踪器

#### 1. 定义一个跟踪器

创建一个新文件 `mmtrack/models/mot/trackers/my_tracker.py`。`BaseTracker` 是提供跨视频跟踪基础 APIs，我们推荐新的跟踪器继承该类，用户可以参考 [BaseTracker](https://github.com/open-mmlab/mmtracking/tree/master/mmtrack/models/mot/trackers/base_tracker.py) 的文档来了解细节。

```python
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker

@TRACKERS.register_module()
class MyTracker(BaseTracker):

    def __init__(self,
                 arg1,
                 arg2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def track(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引入模块

你可以在 `mmtrack/models/mot/trackers/__init__.py` 中加入下面一行。

```python
from .my_tracker import MyTracker
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.mot.trackers.my_tracker.py'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config文件

```python
tracker=dict(
    type='MyTracker',
    arg1=xxx,
    arg2=xxx)
```

### 增加一个新的检测器

请参考[MMDetection教程](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_models.html)来开发新检测器

### 增加一个新的运动估计器

#### 1. 定义一个运动估计模型（例如：MyFlowNet）

新建一个文件 `mmtrack/models/motion/my_flownet.py`。

如果该运动估计模型是一个深度学习模块，你可以继承 `mmcv.runner` 的 `BaseModule`，否则继承 `Object`。

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

### 增加一个新的重识别模型

#### 1. 定义一个识别模型（例如：MyReID）

新建一个文件 `mmtrack/models/motion/my_flownet.py`。

```python
from mmcv.runner import BaseModule

from ..builder import REID

@REID.register_module()
class MyReID(BaseModule):

    def __init__(self,
                arg1,
                arg2):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引入模块

你可以在 `mmtrack/models/reid/__init__.py` 中加入下面一行。

```python
from .my_reid import MyReID
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.reid.my_reid.py'],
    allow_failed_imports=False)

```

#### 3. 更改原始 config 文件

```python
motion=dict(
    type='MyReID',
    arg1=xxx,
    arg2=xxx)
```

### 增加一个新的跟踪头

#### 1. 定义一个跟踪头（例如：MyHead）

新建一个文件 `mmtrack/models/track_heads/my_head.py`。

```python
from mmcv.runner import BaseModule

from mmdet.models import HEADS

@HEADS.register_module()
class MyHead(BaseModule):

    def __init__(self,
                arg1,
                arg2):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引入模块

你可以在 `mmtrack/models/track_heads/__init__.py` 中加入下面一行。

```python
from .my_head import MyHead
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.track_heads.my_head.py'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config 文件

```python
motion=dict(
    type='MyHead',
    arg1=xxx,
    arg2=xxx)
```

### 增加一个新的损失函数

#### 1. 定义一个损失函数

假定你想要增加一个新的损失函数 `MyLoss` 来进行边界框回归。为此，你需要定义一个文件 `mmtrack/models/losses/my_loss.py`。装饰器 `weighted_loss` 可以对损失函数输出结果做基于单个元素的加权平均。

```python
import torch
import torch.nn as nn

from mmdet.models import LOSSES, weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
```

#### 2. 引入模块

你可以在 `mmtrack/models/losses/__init__.py` 中加入下面一行。

```python
from .my_loss import MyLoss, my_loss
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports=dict(
    imports=['mmtrack.models.losses.my_loss'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config 文件

为了使用新的损失函数，你需要更改 `loss_xxx` 区域。
假设 `MyLoss` 是用于回归任务，则需在 `head` 区域中更改 `loss_bbox`。

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
