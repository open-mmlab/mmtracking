## Introduction

MMTracking is an open source video perception toolbox by [PyTorch](https://pytorch.org/). It is a part of [OpenMMLab](https://openmmlab.com) project.

It supports 4 video tasks:

- Video object detection (VID)
- Single object tracking (SOT)
- Multiple object tracking (MOT)
- Video instance segmentation (VIS)

## Major features

- **The First Unified Video Perception Platform**

  We are the first open source toolbox that unifies versatile video perception tasks including video object detection, multiple object tracking, single object tracking and video instance segmentation.

- **Modular Design**

  We decompose the video perception framework into different components and one can easily construct a customized method by combining different modules.

- **Simple, Fast and Strong**

  **Simple**: MMTracking interacts with other OpenMMLab projects. It is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) that we can capitalize any detector only through modifying the configs.

  **Fast**: All operations run on GPUs. The training and inference speeds are faster than or comparable to other implementations.

  **Strong**: We reproduce state-of-the-art models and some of them even outperform the official implementations.

## Getting Started

Please refer to [get_started.md](./get_started.md) for the basic usage of MMTracking.

A Colab tutorial is provided. You may preview the notebook [here](../../demo/MMTracking_Tutorial.ipynb) or directly run it on [Colab](https://colab.research.google.com/github/open-mmlab/mmtracking/blob/master/demo/MMTracking_Tutorial.ipynb).

## User Guides

There are some basic [usage guides](./user_guides/), including:

- [configs](./user_guides/1_config.md)
- [dataset preparation](./user_guides/2_dataset_prepare.md)
- [inference](./user_guides/3_inference.md)
- [train and test](./user_guides/4_train_test.md)
- [visualization](./user_guides/5_visualization.md)
- [analysis tools](./user_guides/6_analysis_tools.md)

If you want to learn more [advanced guides](./advanced_guides), you can refer to:

- [data flow](./advanced_guides/1_data_flow.md)
- [structures](./advanced_guides/2_structures.md)
- [models](./advanced_guides/3_models.md)
- [datasets](./advanced_guides/4_datasets.md)
- [transforms](./advanced_guides/5_transforms.md)
- [evaluation](./advanced_guides/6_evaluation.md)
- [engine](./advanced_guides/7_engine.md)
- [convention](./advanced_guides/8_convention.md)
- [add modules](./advanced_guides/9_add_modules.md)
- [add datasets](./advanced_guides/10_add_datasets.md)
- [add transforms](./advanced_guides/11_add_transforms.md)
- [add metrics](./advanced_guides/12_add_metrics.md)
- [customized runtime](./advanced_guides/13_custime_runtime.md)

## Benchmark and model zoo

Results and models are available in the [model zoo](./model_zoo.md).

## Contributing

We appreciate all contributions to improve MMTracking. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline and [this discussion](https://github.com/open-mmlab/mmtracking/issues/73) for development roadmap.

## FAQ

If you encounter any problems in the process of using MMTracking, you can firstly refer to [FAQ](./notes/faq.md). If not solved, you can post an [issue](https://github.com/open-mmlab/mmtracking/issues/) and we will give a response as soon as possible.
