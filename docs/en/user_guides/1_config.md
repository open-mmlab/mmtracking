# Learn about Configs

We use python files as our config system. You can find all the provided configs under $MMTracking/configs.

We incorporate modular and inheritance design into our config system,
which is convenient to conduct various experiments.
If you wish to inspect the config file,
you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

## A brief description of a complete config

A complete config usually contains the following primary fields:

- `model`: the basic config of model, which may contain `data_preprocessor`, modules (e.g., `detector`, `motion`),`train_cfg`, `test_cfg`, etc.
- `train_dataloader`: the config of training dataloader, which usually contains `batch_size`, `num_workers`, `sampler`, `dataset`, etc.
- `val_dataloader`: the config of validation dataloader, which is similar with `train_dataloader`.
- `test_dataloader`: the config of testing dataloader, which is similar with `train_dataloader`.
- `val_evaluator`: the config of validation evaluator. For example, `type='CocoVideoMetric'` for VID task on the ILSVRC benchmark, `type='MOTChallengeMetrics'` for MOT task on the MOTChallenge benchmarks.
- `test_evaluator`: the config of testing evaluator, which is similar with `val_evaluator`.
- `train_cfg`: the config of training loop. For example, `type='EpochBasedTrainLoop'`.
- `val_cfg`: the config of validation loop. For example, `type='ValLoop'`.
- `test_cfg`: the config of testing loop. For example, `type='TestLoop'`.
- `default_hooks`: the config of default hooks, which may include hooks for timer, logger, param_scheduler, checkpoint, sampler_seed, visualization, etc.
- `vis_backends`: the config of visualization backends, which uses `type='LocalVisBackend'` as default.
- `visualizer`: the config of visualizer. For example, `type='DetLocalVisualizer'` for VID task, and `type='TrackLocalVisualizer'` for MOT, VIS, SOT, VOS tasks.
- `param_scheduler`: the config of parameter scheduler, which usually sets the learning rate scheduler.
- `optim_wrapper`: the config of optimizer wrapper, which contains optimization-related information, for example optimizer, gradient clipping, etc.
- `load_from`: load models as a pre-trained model from a given path.
- `resume`: If `True`, resume checkpoints from `load_from`, and the training will be resumed from the epoch when the checkpoint is saved.

## Modify config through script arguments

When submitting jobs using `tools/train.py` or `tools/test.py`,
you may specify `--cfg-options` to in-place modify the config.
We present several examples as follows.
For more details, please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/ca282aee9e402104b644494ca491f73d93a9544f/mmengine/config/config.py#L919).

- **Update config keys of dict chains.**

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.detector.backbone.norm_eval=False` changes the all BN modules in model backbones to train mode.

- **Update keys inside a list of configs.**

  Some config dicts are composed as a list in your config.
  For example, the testing pipeline `test_dataloader.dataset.pipeline` is normally a list e.g. `[dict(type='LoadImageFromFile'), ...]`.
  If you want to change `LoadImageFromFile` to `LoadImageFromWebcam` in the pipeline,
  you may specify `--cfg-options test_dataloader.dataset.pipeline.0.type=LoadImageFromWebcam`.

- **Update values of list/tuples.**

  Maybe the value to be updated is a list or a tuple.
  For example, you can change the key `mean` of `data_preprocessor` by specifying `--cfg-options model.data_preprocessor.mean=[0,0,0]`.
  Note that **NO** white space is allowed inside the specified value.

## Config File Structure

There are 3 basic component types under `config/_base_`, i.e., dataset, model and default_runtime.
Many methods could be easily constructed with one of each like DFF, FGFA, SELSA, SORT, DeepSORT.
The configs that are composed by components from `_base_` are called *primitive*.

For all configs under the same folder, it is recommended to have only **one** *primitive* config.
All other configs should inherit from the *primitive* config.
In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from exiting methods.
For example, if some modification is made base on Faster R-CNN,
user may first inherit the basic Faster R-CNN structure
by specifying `_base_ = ../../_base_/models/faster-rcnn_r50-dc5.py`,
then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods,
you may create a folder `method_name` under `configs`.

Please refer to [mmengine](https://mmengine.readthedocs.io/en/latest/understand_mmengine/config.html) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```shell
{method}_{module}_{train_cfg}_{train_data}_{test_data}
```

- `{method}`: method name, like `dff`, `deepsort`, `siamese_rpn`.
- `{module}`: basic modules of the method, like `faster-rcnn_r50_fpn`.
- `{train_cfg}`: training config which usually contains batch size, epochs, etc, like `8xb4-80e`.
- `{train_data}`: training data, like `mot17halftrain`.
- `{test_data}`: testing data, like `test-mot17halfval`.

## FAQ

**Ignore some fields in the base configs**

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [mmengine](https://mmengine.readthedocs.io/en/latest/understand_mmengine/config.html) for simple illustration.

**Use intermediate variables in configs**

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs,
user need to pass the intermediate variables into corresponding fields again.
For example, we would like to use testing strategy of adaptive stride to test SELSA.
ref_img_sampler is intermediate variable we would like to modify.

```python
_base_ = ['./selsa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py']
# dataset settings
ref_img_sampler=dict(
    _delete_=True,
    num_ref_imgs=14,
    frame_range=[-7, 7],
    method='test_with_adaptive_stride')
val_dataloader = dict(
    dataset=dict(ref_img_sampler=ref_img_sampler))
test_dataloader = dict(
    dataset=dict(ref_img_sampler=ref_img_sampler))
```

We first define the new `ref_img_sampler` and pass them into `val_dataloader` and `test_dataloader`.
