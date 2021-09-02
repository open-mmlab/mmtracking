## 自定义单目标跟踪模型

我们通常将模型组件分为4类：

- 主干网络：通常是一个用于抽取特征图的FCN网络，例如：ResNet, MobileNet。
- 模型颈部：通常是连接骨干网络和模型头部的组件，例如：ChannelMapper,FPN。
- 模型头部：用于特定任务的组件，例如：跟踪框预测。
- 损失函数：计算损失函数的部件，例如：FocalLoss, L1Loss。

### 增加一个新的主干网络

这里，我们以 MobileNet 为例来展示如何开发一个新组件。

#### 1. 定义一个新主干网络 （例如：MobileNet）

创建一个新文件 `mmtrack/models/backbones/mobilenet.py`

```python
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class MobileNet(BaseModule):

    def __init__(self, arg1, arg2, *args, **kwargs):
        pass

    def forward(self, x):  # should return a tuple
        pass
```

#### 2. 引进模块

你可以在 `mmtrack/models/backbones/__init__.py` 增加下面一行

```python
from .mobilenet import MobileNet
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.backbones.mobilenet'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config 文件

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### 增加一个新的模型瓶颈

#### 1. 定义一个模型瓶颈 （例如：MyFPN）

创建一个新文件 `mmtrack/models/necks/my_fpn.py`

```python
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS

@NECKS.register_module()
class MyFPN(BaseModule):

    def __init__(self, arg1, arg2, *args, **kwargs):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引进模块

你可以在 `mmtrack/models/necks/__init__.py` 增加下面一行

```python
from .my_fpn import MyFPN
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.models.necks.my_fpn.py'],
    allow_failed_imports=False)
```

#### 3. 更改原始 config 文件

```python
neck=dict(
    type='MyFPN',
    arg1=xxx,
    arg2=xxx),
```

### 增加一个新的模型头部

#### 1. 定义一个模型头部 （例如：MyHead）

创建一个新文件 `mmtrack/models/track_heads/my_head.py`

```python
from mmcv.runner import BaseModule

from mmdet.models import HEADS

@HEADS.register_module()
class MyHead(BaseModule):

    def __init__(self, arg1, arg2, *args, **kwargs):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

#### 2. 引进模块

你可以在 `mmtrack/models/track_heads/__init__.py` 增加下面一行

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
track_head=dict(
    type='MyHead',
    arg1=xxx,
    arg2=xxx)
```

### 增加一个新的损失函数

详细请参考 [增加新损失函数](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/customize_mot_model.html#add-a-new-loss)
