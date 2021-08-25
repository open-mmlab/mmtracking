## Customize SOT Models

We basically categorize model components into 4 types.

- backbone: usually an FCN network to extract feature maps, e.g., ResNet, MobileNet.
- neck: the component between backbones and heads, e.g., ChannelMapper, FPN.
- head: the component for specific tasks, e.g., tracking bbox prediction.
- loss: the component in head for calculating losses, e.g., FocalLoss, L1Loss.

### Add a new backbone

Here we show how to develop new components with an example of MobileNet.

#### 1. Define a new backbone (e.g. MobileNet)

Create a new file `mmtrack/models/backbones/mobilenet.py`.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/backbones/__init__.py`

```python
from .mobilenet import MobileNet
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.backbones.mobilenet'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

#### 3. Use the backbone in your config file

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

### Add a new neck

#### 1. Define a neck (e.g. MyFPN)

Create a new file `mmtrack/models/necks/my_fpn.py`.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/necks/__init__.py`,

```python
from .my_fpn import MyFPN
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.necks.my_fpn.py'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Modify the config file

```python
neck=dict(
    type='MyFPN',
    arg1=xxx,
    arg2=xxx),
```

### Add a new head

#### 1. Define a head (e.g. MyHead)

Create a new file `mmtrack/models/track_heads/my_head.py`.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/track_heads/__init__.py`,

```python
from .my_head import MyHead
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.track_heads.my_head.py'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Modify the config file

```python
track_head=dict(
    type='MyHead',
    arg1=xxx,
    arg2=xxx)
```

### Add a new loss

Please refer to [Add a new loss](https://mmtracking.readthedocs.io/en/latest/tutorials/customize_mot_model.html#add-a-new-loss) for developping a new loss.
