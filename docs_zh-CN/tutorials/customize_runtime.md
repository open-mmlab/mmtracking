## 自定义训练配置

### 自定义优化设置

#### 自定义 Pytorch 中的优化器

我们已经支持使用 Pytorch 所有的优化器，并且仅在 config 文件中更改设置即可使用。例如，如果你使用 `ADAM`, 更改如下：

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

为了更改模型的学习率，使用者只需要更改 config 文件中的优化器的 `lr` 参数。使用者可以直接按照 Pytorch 的 [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)设置参数。

#### 自定义自己实现的优化器

#### 1. 定义一个新的优化器

一个自定义的优化器如下：

假定你增加的优化器为 `MyOptimizer`, 它有参数 a, b, c。你需要建立一个新的文件 `mmtrack/core/optimizer/my_optimizer.py`。

```python
from torch.optim import Optimizer
from mmcv.runner.optimizer import OPTIMIZERS


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)
```

#### 2. 将优化器加入注册器中

为了能够找到上述定义的模块，它应首先被引入到主命名空间中。我们提供两种方式来实现：

- 在文件 `mmtrack/core/optimizer/__init__.py` 中引入

新定义的 module 应被引入到 `mmtrack/core/optimizer/__init__.py`， 以便注册器能够发现该新模块并添加它。

```python
from .my_optimizer import MyOptimizer
```

- 在 config 文件中使用 `custom_imports` 来手动的引用它

```python
 custom_imports = dict(imports=['mmtrack.core.optimizer.my_optimizer.py'], allow_failed_imports=False)
```

在项目开始阶段，模块 `mmtrack.core.optimizer.my_optimizer.MyOptimizer` 将会被引入， `MyOptimizer` 类会被自动注册。注意：我们引入的应该是只包含 `MyOptimizer` 的文件，而不是直接像这样 `mmtrack.core.optimizer.my_optimizer.MyOptimizer` 引入该类。

实际上使用者也可以将该模块定义在别的文件目录，只要该模块所在目录在 `PYTHONPATH` 里面能被找到。

#### 3.在 config 文件中指定优化器

你可以在 config 的`optimizer` 区域使用 `MyOptimizer`。在 config 文件中，原始优化器定义如下：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

为了使用你自己的优化器，该区域可以更改如下：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

#### 自定义优化器构建器

有的模型有一些用于模型优化的特定参数，例如：用于批归一化层的权值衰减。使用者可以通过自定义优化器构建器来调节这些参数。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmtrack.utils import get_root_logger
from .my_optimizer import MyOptimizer


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(object):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer
```

默认的优化器构建器[在此](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.DefaultOptimizerConstructor), 它可作为新优化器构建器的模板。

#### 额外的设置

没有被优化器实现的技巧应该被优化器配置器(例如：参数特定学习率)或者钩子实现。我们列举了一些常用的可以稳定训练或者加速训练的设置。大家可以自由提出PR、issue来得到更多的设置。

- __使用梯度裁剪来稳定训练__：一些模型需要梯度裁剪来稳定训练过程。如下所示：

```python
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
```

如果你的 config 继承于基础 config, 并且已经设置了优化器基础 config, 你需要设置 `_delete=True` 来重写不必要的设置。 详情请见 [config 文档](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields)

- __使用动量调度器来加快模型收敛__。我们支持动量调度器来根据学习率改变模型的动量，使模型以更快的方式收敛。动量调度器通常和学习率调度器一起使用，例如：下面在 3D 检测中使用的 config 来加速模型收敛。 更多细节请参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327) and [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130)。

```python
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
```

### 自定义训练调度器

我们支持多种[学习率调整策略](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)， 例如 `CosineAnnealing` 和 `Poly`。如下所示：

- `Poly` 调度器

```python
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
```

- `CosineAnnealing` 调度器

```python
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
```

### 自定义工作流

工作流是一个包含（阶段名，迭代轮数）的列表，用来指定运行顺序和迭代轮数。默认设置如下：

```python
workflow = [('train', 1)]
```

上述代码表示执行一轮训练阶段。有时，使用者可能想要检查模型在验证集上的一些度量指标 (例如：损失函数和准确度)。这种情况下，我们设置如下：

```python
[('train', 1), ('val', 1)]
```

这样，经过一轮训练阶段后，将执行一次验证阶段。

**注意**

1. 模型的参数将不会在验证阶段更新。

2. config 中的关键字 `total_epoch` 是用来控制训练阶段轮数，不影响验证阶段。

3. 工作流 `[('train',1),('val',1)]` 和 `[('train',1)]` 将不会改变 `EvalHook`，因为 `EvalHook` 是被 `after_train_epoch` 调用，验证工作流仅影响在 `after_val_epoch` 中调用的钩子。因此，`[('train',1),('val',1)]` 和 `[('train',1)]` 唯一的不同就是 runner 将在每个训练循环结束后计算验证集上的损失函数。

### 自定义钩子

#### 自定义自己实现的钩子

#### 1. 实现一个新的钩子

在很多情况下，使用者需要实现一个新的钩子。MMTracking 支持在训练阶段自定义钩子。因此，使用者可以在 mmtrack 或者基于 mmtrack 的代码库中，通过修改训练 config 定义一个钩子。这里我们给出一个在 mmtrack 中创建一个钩子，并在 training 中使用它的范例：

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

根据钩子的功能，用户需要指定在训练的每个阶段 `before_run`, `after_run`, `before_epoch`, `after_epoch`, `before_iter`, and `after_iter` 钩子执行的事情。

#### 2. 注册一个新的钩子

假定你需要注册的钩子为 `MyHook`，你可以在 `mmtrack/core/utils/__init__.py` 增加下面一行

```python
from .my_hook import MyHook
```

或者，为了避免更改原始代码，你还可以在 config 文件中增加以下几行来实现：

```python
custom_imports = dict(
    imports=['mmtrack.core.utils.my_hook'],
    allow_failed_imports=False)
```

#### 3. 更改 config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

你也可以通过设置关键词 `priority` 为 `NORMAL` 或者 `HIGHEST` 来设置钩子的优先级:

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

在注册器中钩子的默认优先级为 `NORMAL`。

#### 使用 MMCV 中已定义的钩子

如果钩子已经在 MMCV 中定义实现过，你可以直接在 config 文件中添加：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

#### 修改默认的钩子

有一些常用的钩子不是通过 `custom_hooks` 来注册，他们包括：

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

在这些钩子中，只有日志钩子是 `VERY_LOW` 优先级，其余的优先级为 `NORMAL`。上述的教程已经包括如何修改 `optimizer_config`, `momentum_config` 和 `lr_config`。这里我们展示可以用 `log_config`、`checkpoint_config` 和 `evaluation` 做的事情。

#### 模型保存钩子

MMCV 中的 runner 使用 `checkpoint_config` 来初始化[`模型保存钩子`](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

```python
checkpoint_config = dict(interval=1)
```

使用者可以设置 `max_keep_ckpts` 来仅保存一部分历史模型，可以设置 `save_optimizer` 来决定是否保存优化器参数。更多细节请参考[`模型保存钩子`](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

#### 日志钩子

`log_config` 包括多个日志钩子，并可以设置间隔时间。目前MMCV支持 `WandbLoggerHook`、`MlflowLoggerHook`、`TensorboardLoggerHook`。具体使用可以参考[日志钩子](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook)

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 评估钩子

config 中的 `evaluation` 将被用来初始化[`评估钩子`](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.EvalHook)。除了 `intertal`、`start` 等关键词外，其他参数例如 `metric` 将被传入 `dataset.evaluate()`

```python
evaluation = dict(interval=1, metric='bbox')
```
