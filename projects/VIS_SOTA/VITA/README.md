# VITA: Video Instance Segmentation via Object Token Association

## Description

This is an implementation of [VITA](https://github.com/sukjunhwang/VITA) based on [MMTracking](https://github.com/open-mmlab/mmtracking/tree/1.x), [MMDetection](https://github.com/open-mmlab/mmdetection/tree/3.x), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

We introduce a novel paradigm for offline Video Instance Segmentation (VIS), based on the hypothesis that explicit object-oriented information can be a strong clue for understanding the context of the entire sequence. To this end, we proposeVITA, a simple structure built on top of an off-the-shelf Transformer-based image instance segmentation model. Specifically, we use an image object detector as a means of distilling object-specific contexts into object tokens. VITA accomplishes video-level understanding by associating frame-level object tokens without using spatio-temporal backbone features. By effectively building relationships between objects using the condensed information, VITA achieves the state-of-the-art on VIS benchmarks with a ResNet-50 backbone: 49.8 AP, 45.7 AP on YouTube-VIS 2019 & 2021 and 19.6 AP on OVIS. Moreover, thanks to its object token-based structure that is disjoint from the backbone features, VITA shows several practical advantages that previous offline VIS methods have not explored - handling long and high-resolution videos with a common GPU and freezing a frame-level detector trained on image domain.

<center>
<img src="https://github.com/sukjunhwang/VITA/blob/main/vita_teaser.png">
</center>

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMTracking's root directory, run the following command to train the model:

```bash
python tools/train.py projects/VIS_SOTA/IDOL/configs/vita_r50_8xb2-8e_youtubevis2019.py
```

For multi-gpu training, run:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/VIS_SOTA/IDOL/configs/vita_r50_8xb2-8e_youtubevis2019.py
```

### Testing commands

In MMTracking's root directory, run the following command to test the model:

```bash
python tools/test.py projects/VIS_SOTA/IDOL/configs/vita_r50_8xb2-8e_youtubevis2019.py ${CHECKPOINT_PATH}
```

## Results

#### YouTubeVIS-2019

| Method | Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  |                                   Config                                    |         Download         |
| :----: | :------: | :-----: | :-----: | :------: | :------------: | :--: | :-------------------------------------------------------------------------: | :----------------------: |
|  VITA  |   R-50   | pytorch |  140k   |   10.0   |       -        | 50.3 | [config](projects/VIS_SOTA/VITA/configs/vita_r50_8xb2-8e_youtubevis2019.py) | [model](<>) \| [log](<>) |

#### OVIS

| Method | Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  |    Config    |         Download         |
| :----: | :------: | :-----: | :-----: | :------: | :------------: | :--: | :----------: | :----------------------: |
|  VITA  |   R-50   | pytorch |  150k   |   10.0   |       -        | 19.2 | [config](<>) | [model](<>) \| [log](<>) |

## Citation

If you find VITA is useful in your research or applications, please consider giving a star 🌟 to the [official repository](https://github.com/sukjunhwang/VITA) and citing VITA by the following BibTeX entry.

```BibTeX
@inproceedings{VITA,
  title={VITA: Video Instance Segmentation via Object Token Association},
  author={Heo, Miran and Hwang, Sukjun and Oh, Seoung Wug and Lee, Joon-Young and Kim, Seon Joo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/mmdet/datasets/transforms/transforms.py#L41-L169) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/tests/test_datasets/test_transforms/test_transforms.py#L35-L88) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
