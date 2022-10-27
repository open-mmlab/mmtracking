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

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmtrack)](https://pypi.org/project/mmtrack/)
[![PyPI](https://img.shields.io/pypi/v/mmtrack)](https://pypi.org/project/mmtrack)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmtracking.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmtracking/workflows/build/badge.svg)](https://github.com/open-mmlab/mmtracking/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmtracking/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmtracking)
[![license](https://img.shields.io/github/license/open-mmlab/mmtracking.svg)](https://github.com/open-mmlab/mmtracking/blob/master/LICENSE)

[📘Documentation](https://mmtracking.readthedocs.io/) |
[🛠️Installation](https://mmtracking.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmtracking.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmtracking.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmtracking/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMTracking is an open source video perception toolbox by PyTorch. It is a part of [OpenMMLab](https://openmmlab.com) project.

The master branch works with **PyTorch1.5+**.

<div align="center">
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

## What's New

We release MMTracking 1.0.0rc0, the first version of MMTracking 1.x.

Built upon the new [training engine](https://github.com/open-mmlab/mmengine), MMTracking 1.x unifies the interfaces of datasets, models, evaluation, and visualization.

We also support more methods in MMTracking 1.x, such as [StrongSORT](https://github.com/open-mmlab/mmtracking/tree/dev-1.x/configs/mot/strongsort) for MOT, [Mask2Former](https://github.com/open-mmlab/mmtracking/tree/dev-1.x/configs/vis/mask2former) for VIS, [PrDiMP](https://github.com/open-mmlab/mmtracking/tree/dev-1.x/configs/sot/prdimp) for SOT.

Please refer to [dev-1.x](https://github.com/open-mmlab/mmtracking/tree/dev-1.x) branch for the using of MMTracking 1.x.

## Installation

Please refer to [install.md](docs/en/install.md) for install instructions.

## Getting Started

Please see [dataset.md](docs/en/dataset.md) and [quick_run.md](docs/en/quick_run.md) for the basic usage of MMTracking.

A Colab tutorial is provided. You may preview the notebook [here](./demo/MMTracking_Tutorial.ipynb) or directly run it on [Colab](https://colab.research.google.com/github/open-mmlab/mmtracking/blob/master/demo/MMTracking_Tutorial.ipynb).

There are also usage [tutorials](docs/en/tutorials/), such as [learning about configs](docs/en/tutorials/config.md), [an example about detailed description of vid config](docs/en/tutorials/config_vid.md), [an example about detailed description of mot config](docs/en/tutorials/config_mot.md), [an example about detailed description of sot config](docs/en/tutorials/config_sot.md), [customizing dataset](docs/en/tutorials/customize_dataset.md), [customizing data pipeline](docs/en/tutorials/customize_data_pipeline.md), [customizing vid model](docs/en/tutorials/customize_vid_model.md), [customizing mot model](docs/en/tutorials/customize_mot_model.md), [customizing sot model](docs/en/tutorials/customize_sot_model.md), [customizing runtime settings](docs/en/tutorials/customize_runtime.md) and [useful tools](docs/en/useful_tools_scripts.md).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

### Video Object Detection

Supported Methods

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)

Supported Datasets

- [x] [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)

### Single Object Tracking

Supported Methods

- [x] [SiameseRPN++](configs/sot/siamese_rpn) (CVPR 2019)
- [x] [STARK](configs/sot/stark) (ICCV 2021)
- [x] [MixFormer](configs/sot/mixformer) (CVPR 2022)
- [ ] [PrDiMP](https://arxiv.org/abs/2003.12565) (CVPR2020) (WIP)

Supported Datasets

- [x] [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
- [x] [UAV123](https://cemse.kaust.edu.sa/ivul/uav123/)
- [x] [TrackingNet](https://tracking-net.org/)
- [x] [OTB100](http://www.visual-tracking.net/)
- [x] [GOT10k](http://got-10k.aitestunion.com/)
- [x] [VOT2018](https://www.votchallenge.net/vot2018/)

### Multi-Object Tracking

Supported Methods

- [x] [SORT/DeepSORT](configs/mot/deepsort) (ICIP 2016/2017)
- [x] [Tracktor](configs/mot/tracktor) (ICCV 2019)
- [x] [QDTrack](configs/mot/qdtrack) (CVPR 2021)
- [x] [ByteTrack](configs/mot/bytetrack) (ECCV 2022)
- [x] [OC-SORT](configs/mot/ocsort) (arXiv 2022)

Supported Datasets

- [x] [MOT Challenge](https://motchallenge.net/)
- [x] [CrowdHuman](https://www.crowdhuman.org/)
- [x] [LVIS](https://www.lvisdataset.org/)
- [x] [TAO](https://taodataset.org/)
- [x] [DanceTrack](https://arxiv.org/abs/2111.14690)

### Video Instance Segmentation

Supported Methods

- [x] [MaskTrack R-CNN](configs/vis/masktrack_rcnn) (ICCV 2019)

Supported Datasets

- [x] [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

## Contributing

We appreciate all contributions to improve MMTracking. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline and [this discussion](https://github.com/open-mmlab/mmtracking/issues/73) for development roadmap.

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

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
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
