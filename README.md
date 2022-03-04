<div align="center">
  <img src="resources/mmtrack-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
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

English | [简体中文](/README_zh-CN.md)

Documentation: https://mmtracking.readthedocs.io/

## Introduction

MMTracking is an open source video perception toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The master branch works with **PyTorch1.5+**.

<div align="left">
  <img src="https://user-images.githubusercontent.com/24663779/103343312-c724f480-4ac6-11eb-9c22-b56f1902584e.gif" width="800"/>
</div>

### Major features

- **The First Unified Video Perception Platform**

  We are the first open source toolbox that unifies versatile video perception tasks include video object detection, multiple object tracking, single object tracking and video instance segmentation.

- **Modular Design**

  We decompose the video perception framework into different components and one can easily construct a customized method by combining different modules.

- **Simple, Fast and Strong**

  **Simple**: MMTracking interacts with other OpenMMLab projects. It is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) that we can capitalize any detector only through modifying the configs.

  **Fast**: All operations run on GPUs. The training and inference speeds are faster than or comparable to other implementations.

  **Strong**: We reproduce state-of-the-art models and some of them even outperform the official implementations.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

Release [STARK](configs/sot/stark) pretrained models.

v0.11.0 was released in 04/03/2022.
Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

Supported methods of video object detection:

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)

Supported methods of multi object tracking:

- [x] [SORT/DeepSORT](configs/mot/deepsort) (ICIP 2016/2017)
- [x] [Tracktor](configs/mot/tracktor) (ICCV 2019)
- [x] [ByteTrack](configs/mot/bytetrack) (arXiv 2021)

Supported methods of single object tracking:

- [x] [SiameseRPN++](configs/sot/siamese_rpn) (CVPR 2019)
- [x] [STARK](configs/sot/stark) (ICCV 2021)

Supported methods of video instance segmentation:

- [x] [MaskTrack R-CNN](configs/vis/masktrack_rcnn) (ICCV 2019)

## Installation

Please refer to [install.md](docs/en/install.md) for install instructions.

## Getting Started

Please see [dataset.md](docs/en/dataset.md) and [quick_run.md](docs/en/quick_run.md) for the basic usage of MMTracking.
We also provide usage [tutorials](docs/en/tutorials/), such as [learning about configs](docs/en/tutorials/config.md), [an example about detailed description of vid config](docs/en/tutorials/config_vid.md), [an example about detailed description of mot config](docs/en/tutorials/config_mot.md), [an example about detailed description of sot config](docs/en/tutorials/config_sot.md), [customizing dataset](docs/en/tutorials/customize_dataset.md), [customizing data pipeline](docs/en/tutorials/customize_data_pipeline.md), [customizing vid model](docs/en/tutorials/customize_vid_model.md), [customizing mot model](docs/en/tutorials/customize_mot_model.md), [customizing sot model](docs/en/tutorials/customize_sot_model.md), [customizing runtime settings](docs/en/tutorials/customize_runtime.md) and [useful tools](docs/en/useful_tools_scripts.md).

## Contributing

We appreciate all contributions to improve MMTracking. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMTracking is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new video perception methods.

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmtrack2020,
    title={{MMTracking: OpenMMLab} video perception toolbox and benchmark},
    author={MMTracking Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmtracking}},
    year={2020}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration):  OpenMMLab Generative Model toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMlab deep learning model deployment toolset.
