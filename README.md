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
[![docs](https://img.shields.io/badge/docs-1.x-blue)](https://mmtracking.readthedocs.io/en/1.x/)
[![badge](https://github.com/open-mmlab/mmtracking/workflows/build/badge.svg)](https://github.com/open-mmlab/mmtracking/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmtracking/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmtracking)
[![license](https://img.shields.io/github/license/open-mmlab/mmtracking.svg)](https://github.com/open-mmlab/mmtracking/blob/master/LICENSE)

[📘Documentation](https://mmtracking.readthedocs.io/en/1.x/) |
[🛠️Installation](https://mmtracking.readthedocs.io/en/1.x/get_started.html) |
[👀Model Zoo](https://mmtracking.readthedocs.io/en/1.x/model_zoo.html) |
[🆕Update News](https://mmtracking.readthedocs.io/en/1.x/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmtracking/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

## Introduction

MMTracking is an open source video perception toolbox by PyTorch. It is a part of [OpenMMLab](https://openmmlab.com) project.

The master branch works with **PyTorch1.6+**.

<div align="center">
  <img src="https://user-images.githubusercontent.com/24663779/103343312-c724f480-4ac6-11eb-9c22-b56f1902584e.gif" width="800"/>
</div>

### Major features

- **The First Unified Video Perception Platform**

  We are the first open source toolbox that unifies versatile video perception tasks include video object detection, multiple object tracking, single object tracking and video instance segmentation.

- **Modular Design**

  We decompose the video perception framework into different components and one can easily construct a customized method by combining different modules.

- **Simple, Fast and Strong**

  **Simple**: MMTracking interacts with other OpenMMLab projects. It is built upon [MMDetection](https://github.com/open-mmlab/mmdetection/tree/3.x) that we can capitalize any detector only through modifying the configs.

  **Fast**: All operations run on GPUs. The training and inference speeds are faster than or comparable to other implementations.

  **Strong**: We reproduce state-of-the-art models and some of them even outperform the official implementations.

## What's New

Release [Mask2Former](configs/vis/mask2former), [PrDiMP](configs/sot/prdimp) and [StrongSORT++](configs/mot/strongsort)) pretrained models.

v1.0.0rc1 was released in 10/10/2022.
Please refer to [changelog.md](docs/en/notes/changelog.md) for details and release history.

## Get Started

Please refer to [get_started.md](docs/en/get_started.md) for install instructions.

Please refer to [inference.md](docs/en/user_guides/3_inference.md) for the basic usage of MMTracking. If you want to train and test your own model, please see [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md) and [train_test.md](docs/en/user_guides/4_train_test.md).

A Colab tutorial is also provided. You may preview the notebook [here](./demo/MMTracking_Tutorial.ipynb) or directly run it on [Colab](https://colab.research.google.com/github/open-mmlab/mmtracking/blob/master/demo/MMTracking_Tutorial.ipynb).

There are also usage [tutorials](docs/en/user_guides/), such as [learning about configs](docs/en/user_guides/1_config.md), [visualization](docs/en/user_guides/5_visualization.md), [analysis tools](docs/en/user_guides/6_analysis_tools.md),

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

### Video Object Detection

Supported Methods

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)

Supported Datasets

- [x] [ILSVRC](http://image-net.org/challenges/LSVRC/2015/)

### Multi-Object Tracking

Supported Methods

- [x] [SORT](configs/mot/sort) (ICIP 2016)
- [x] [DeepSORT](configs/mot/deepsort) (ICIP 2017)
- [x] [Tracktor](configs/mot/tracktor) (ICCV 2019)
- [x] [QDTrack](configs/mot/qdtrack) (CVPR 2021)
- [x] [ByteTrack](configs/mot/bytetrack) (ECCV 2022)
- [x] [StrongSORT](configs/mot/strongsort) (arxiv 2022)

Supported Datasets

- [x] [MOT Challenge](https://motchallenge.net/)
- [x] [CrowdHuman](https://www.crowdhuman.org/)
- [x] [LVIS](https://www.lvisdataset.org/)
- [x] [TAO](https://taodataset.org/)
- [x] [DanceTrack](https://arxiv.org/abs/2111.14690)

### Video Instance Segmentation

Supported Methods

- [x] [MaskTrack R-CNN](configs/vis/masktrack_rcnn) (ICCV 2019)
- [x] [Mask2Former](configs/vis/mask2former) (CVPR 2022)

Supported Datasets

- [x] [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

### Single Object Tracking

Supported Methods

- [x] [SiameseRPN++](configs/sot/siamese_rpn) (CVPR 2019)
- [x] [PrDiMP](configs/sot/prdimp) (CVPR2020)
- [x] [STARK](configs/sot/stark) (ICCV 2021)

Supported Datasets

- [x] [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
- [x] [UAV123](https://cemse.kaust.edu.sa/ivul/uav123/)
- [x] [TrackingNet](https://tracking-net.org/)
- [x] [OTB100](http://www.visual-tracking.net/)
- [x] [GOT10k](http://got-10k.aitestunion.com/)
- [x] [VOT2018](https://www.votchallenge.net/vot2018/)

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

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
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
