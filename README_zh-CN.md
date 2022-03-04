<div align="center">
  <img src="resources/mmtrack-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmtrack)](https://pypi.org/project/mmtrack/)
[![PyPI](https://img.shields.io/pypi/v/mmtrack)](https://pypi.org/project/mmtrack)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmtracking.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmtracking/workflows/build/badge.svg)](https://github.com/open-mmlab/mmtracking/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmtracking/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmtracking)
[![license](https://img.shields.io/github/license/open-mmlab/mmtracking.svg)](https://github.com/open-mmlab/mmtracking/blob/master/LICENSE)

[English](/README.md) | 简体中文

文档：https://mmtracking.readthedocs.io/

## 简介

MMTracking是一款基于PyTorch的视频目标感知开源工具箱，是[OpenMMLab](http://openmmlab.org/)项目的一部分。

主分支代码目前支持**PyTorch 1.5以上**的版本。

<div align="left">
  <img src="https://user-images.githubusercontent.com/24663779/103343312-c724f480-4ac6-11eb-9c22-b56f1902584e.gif" width="800"/>
</div>

### 主要特性

- **首个开源一体化视频目标感知平台**

  MMTracking 是首个开源一体化视频目标感知工具箱，同时支持视频目标检测，多目标跟踪，单目标跟踪和视频个例分割等多种任务和算法。

- **模块化设计**

  MMTracking将统一的视频目标感知框架解耦成不同的模块组件，通过组合不同模块组件，用户可以便捷地构建自定义视频目标感知模型。

- **简洁、高效、强大**

  **简洁**：MMTracking与其他OpenMMLab平台充分交互。MMTracking充分复用[MMDetection](https://github.com/open-mmlab/mmdetection)中的已有模块，我们只需要修改配置文件就可以使用任何检测器。

  **高效**：MMTracking所有操作都在GPU上运行。相比其他开源库的实现，MMTracking的训练和推理更加高效。

  **强大**：MMTracking复现了SOTA性能的模型。受益于[MMDetection](https://github.com/open-mmlab/mmdetection)的持续推进，部分实现精度超出官方版本。

## 许可

该项目遵循[Apache 2.0 license](/LICENSE)开源协议。

## 更新记录

v0.11.0版本已于2022年03月04日发布，可通过查阅[更新日志](docs/en/changelog.md)了解更多细节以及发布历史。

## 基准测试与模型库

本工具箱支持的各个模型的结果和设置都可以在[模型库](docs/en/model_zoo.md)页面中查看。

支持的视频目标检测算法:

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)

支持的多目标跟踪算法:

- [x] [SORT/DeepSORT](configs/mot/deepsort) (ICIP 2016/2017)
- [x] [Tracktor](configs/mot/tracktor) (ICCV 2019)
- [x] [ByteTrack](configs/mot/bytetrack) (arXiv 2021)

支持的单目标跟踪算法:

- [x] [SiameseRPN++](configs/sot/siamese_rpn) (CVPR 2019)
- [x] [STARK](configs/sot/stark) (ICCV 2021)

支持的视频实例分割算法:

- [x] [MaskTrack R-CNN](configs/vis/masktrack_rcnn) (ICCV 2019)

## 安装

请参考[安装指南](docs/en/install.md)进行安装。

## 开始使用MMTracking

请参考[数据集](docs/en/dataset.md)和[快速开始](docs/en/quick_run.md)了解MMTracking的基本使用。MMTracking也提供了更详细的[教程](docs/en/tutorials/)，比如[配置文件简介](docs/en/tutorials/config.md), [视频目标检测器配置文件详解](docs/en/tutorials/config_vid.md), [多目标跟踪器配置文件详解](docs/en/tutorials/config_mot.md), [单目标跟踪器配置文件详解](docs/en/tutorials/config_sot.md), [自定义数据集](docs/en/tutorials/customize_dataset.md), [自定义数据预处理流程](docs/en/tutorials/customize_data_pipeline.md), [自定义视频目标检测器](docs/en/tutorials/customize_vid_model.md), [自定义多目标跟踪器](docs/en/tutorials/customize_mot_model.md), [自定义单目标跟踪器](docs/en/tutorials/customize_sot_model.md), [自定义训练配置](docs/en/tutorials/customize_runtime.md) 以及 [有用的工具和脚本](docs/en/useful_tools_scripts.md).

## 参与贡献

我们非常欢迎用户对于MMTracking做出的任何贡献，可以参考[贡献指南](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md)文件了解更多细节。

## 致谢

MMTracking是一款开源项目，我们欢迎任何贡献和反馈。我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己新的视频目标感知方法。

## 引用

如果你觉得MMTracking对你的研究有所帮助，可以考虑引用它:

```latex
@misc{mmtrack2020,
    title={{MMTracking: OpenMMLab} video perception toolbox and benchmark},
    author={MMTracking Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmtracking}},
    year={2020}
}
```

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv)：OpenMMLab计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="https://user-images.githubusercontent.com/24663779/116371114-a8005e80-a83d-11eb-9123-17fc9cfe7475.jpg" height="400" />  <img src="https://user-images.githubusercontent.com/24663779/116371171-b8b0d480-a83d-11eb-9514-247f0e5dbfa3.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
