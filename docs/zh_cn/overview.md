## 简介

MMTracking 是一个基于 [PyTorch](https://pytorch.org/) 的视频感知开源工具箱，它是 [OpenMMLab](https://openmmlab.com) 项目的一部分。

它支持 4 项视频任务:

- 视频目标检测（VID）
- 单目标跟踪 （SOT）
- 多目标跟踪（MOT）
- 视频实例分割（VIS）

## 主要特性

- **首个统一视频感知平台**

  我们是首个统一了多种视频感知任务的开源工具箱，任务包含视频目标检测、视频实例分割、单目标跟踪及多目标跟踪。

- **模块化设计**

  我们将视频感知框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷的构建自定义的模型。

- **简单、快速、强大**

  **简单**: MMTracking 建立在 [MMDetection](https://github.com/open-mmlab/mmdetection) 之上，并且能与 OpenMMLab 上的其他项目交互，通过简单调整配置文件即可使用任何检测算法。

  **快速**: 所有算法操作都运行在 GPU 上，训练速度比其他代码库更快或者相当。

  **强大**: 我们复现了 SOTA 的模型，其中一些模型甚至优于官方实现。

## 入门

MMTracking 的基本使用方法请参考 [get_started.md](./get_started.md) 。

我们提供了 Colab 教程，您可以在[此处](../../demo/MMTracking_Tutorial.ipynb)预览或直接在 [Colab](https://colab.research.google.com/github/open-mmlab/mmtracking/blob/master/demo/MMTracking_Tutorial.ipynb) 上运行。

## 教程

以下是一些[基础教程](./user_guides/)，包含：

- [配置](./user_guides/1_config.md)
- [数据集准备](./user_guides/2_dataset_prepare.md)
- [推理](./user_guides/3_inference.md)
- [训练和测试](./user_guides/4_train_test.md)
- [可视化](./user_guides/5_visualization.md)
- [分析工具](./user_guides/6_analysis_tools.md)

如果您想学习更多[进阶指南](./advanced_guides) ，可参见：

- [数据流](./advanced_guides/1_data_flow.md)
- [结构](./advanced_guides/2_structures.md)
- [模型](./advanced_guides/3_models.md)
- [数据集](./advanced_guides/4_datasets.md)
- [transforms](./advanced_guides/5_transforms.md)
- [评估](./advanced_guides/6_evaluation.md)
- [引擎](./advanced_guides/7_engine.md)
- [convention](./advanced_guides/8_convention.md)
- [添加模块](./advanced_guides/9_add_modules.md)
- [添加数据集](./advanced_guides/10_add_datasets.md)
- [add transforms](./advanced_guides/11_add_transforms.md)
- [添加评价指标](./advanced_guides/12_add_metrics.md)
- [自定义运行环境](./advanced_guides/13_custime_runtime.md)

## 基准测试和模型库

测试结果和模型可以在[模型库](./model_zoo.md)中找到。

## 贡献指南

我们感谢所有为改进 MMTracking 所做出的努力。请参阅 [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) 以获取贡献指南，或[在此](https://github.com/open-mmlab/mmtracking/issues/73)参与开发路线的讨论。

## 常见问题

如果您在使用 MMTracking 过程中遇到任何问题，可以先参考 [FAQ](https://github.com/open-mmlab/mmtracking/blob/dev-1.x/docs/en/notes/faq.md)。如未解决，您可以发布一个 [issue](https://github.com/open-mmlab/mmtracking/issues/)，我们会尽快回复。
