## Customize VID Models

We basically categorize model components into 3 types.

- detector: usually a detector to detect objects from an image, e.g., Faster R-CNN.
- motion: the component to compute motion information between two images, e.g., FlowNetSimple.
- aggregator: the component for aggregating features from multi images, e.g., EmbedAggregator.

### Add a new detector

Please refer to [tutorial in mmdetection](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_models.html) for developping a new detector.

### Add a new motion model

#### 1. Define a motion model (e.g. MyFlowNet)

Create a new file `mmtrack/models/motion/my_flownet.py`.

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

### Add a new aggregator

#### 1. Define a aggregator (e.g. MyAggregator)

Create a new file `mmtrack/models/aggregators/my_aggregator.py`.

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

#### 2. Import the module

You can either add the following line to `mmtrack/models/aggregators/__init__.py`,

```python
from .my_aggregator import MyAggregator
```

or alternatively add

```python
custom_imports = dict(
    imports=['mmtrack.models.aggregators.my_aggregator.py'],
    allow_failed_imports=False)
```

to the config file and avoid modifying the original code.

#### 3. Modify the config file

```python
aggregator=dict(
    type='MyAggregator',
    arg1=xxx,
    arg2=xxx)
```
