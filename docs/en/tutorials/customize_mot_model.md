## Customize MOT Models

We basically categorize model components into 5 types.

- tracker: the component that associate the objects across the video with the cues extracted by the components below.
- detector: usually a detector to detect objects from the input image, e.g., Faster R-CNN.
- motion: the component to compute motion information between consecutive frames, e.g., KalmanFilter.
- reid: usually an independent ReID model to extract the feature embeddings from the cropped image, e.g., BaseReID.
- track_head: the component to extract tracking cues but share the same backbone with the detector, e.g., a embedding head or a regression head.

### Add a new tracker

#### 1. Define a tracker (e.g. MyTracker)

Create a new file `mmtrack/models/mot/trackers/my_tracker.py`.

We implement a `BaseTracker` that provide basic APIs to maintain the tracks across the video.
We recommend to inherit the new tracker from it.
The users may refer to the documentations of [BaseTracker](https://github.com/open-mmlab/mmtracking/tree/master/mmtrack/models/mot/trackers/base_tracker.py) for the details.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/mot/trackers/__init__.py`,

```python
from .my_tracker import MyTracker
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.mot.trackers.my_tracker.py'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Modify the config file

```python
tracker=dict(
    type='MyTracker',
    arg1=xxx,
    arg2=xxx)
```

### Add a new detector

Please refer to [tutorial in mmdetection](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_models.html) for developping a new detector.

### Add a new motion model

#### 1. Define a motion model (e.g. MyFlowNet)

Create a new file `mmtrack/models/motion/my_flownet.py`.

You can inherit the motion model from `BaseModule` in `mmcv.runner` if it is a deep learning module, and from `object` if not.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/motion/__init__.py`,

```python
from .my_flownet import MyFlowNet
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.motion.my_flownet.py'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Modify the config file

```python
motion=dict(
    type='MyFlowNet',
    arg1=xxx,
    arg2=xxx)
```

### Add a new reid model

#### 1. Define a reid model (e.g. MyReID)

Create a new file `mmtrack/models/reid/my_reid.py`.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/reid/__init__.py`,

```python
from .my_reid import MyReID
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.reid.my_reid.py'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Modify the config file

```python
reid=dict(
    type='MyReID',
    arg1=xxx,
    arg2=xxx)
```

### Add a new track head

#### 1. Define a head (e.g. MyHead)

Create a new file `mmtrack/models/track_heads/my_head.py`.

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

#### 1. define a loss (e.g. MyLoss)

Assume you want to add a new loss as `MyLoss`, for bounding box regression.
To add a new loss function, the users need implement it in `mmtrack/models/losses/my_loss.py`.
The decorator `weighted_loss` enable the loss to be weighted for each element.

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

#### 2. Import the module

Then the users need to add it in the `mmtrack/models/losses/__init__.py`.

```python
from .my_loss import MyLoss, my_loss

```

Alternatively, you can add

```python
custom_imports=dict(
    imports=['mmtrack.models.losses.my_loss'],
    allow_failed_imports=False)
```

to the config file and achieve the same goal.

#### 3. Modify the config file

To use it, modify the `loss_xxx` field.
Since MyLoss is for regression, you need to modify the `loss_bbox` field in the `head`.

```python
loss_bbox=dict(type='MyLoss', loss_weight=1.0))
```
