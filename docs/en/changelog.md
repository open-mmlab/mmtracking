## Changelog

### v0.13.0 (29/04/2022)

#### Highlights

- Support tracking colab tutorial ([#511](https://github.com/open-mmlab/mmtracking/pull/511))

#### New Features

- Refactor the training datasets of SiamRPN++ ([#496](https://github.com/open-mmlab/mmtracking/pull/496)), ([#518](https://github.com/open-mmlab/mmtracking/pull/518))

- Support loading data from ceph for SOT datasets ([#494](https://github.com/open-mmlab/mmtracking/pull/494))

- Support loading data from ceph for MOT challenge dataset ([#517](https://github.com/open-mmlab/mmtracking/pull/517))

- Support evaluation metric for VIS task ([#501](https://github.com/open-mmlab/mmtracking/pull/501))

#### Bug Fixes

- Fix a bug in the LaSOT datasets and update the pretrained models of STARK ([#483](https://github.com/open-mmlab/mmtracking/pull/483)), ([#503](https://github.com/open-mmlab/mmtracking/pull/503))

- Fix a bug in the format_results function of VIS task ([#504](https://github.com/open-mmlab/mmtracking/pull/504))

### v0.12.0 (01/04/2022)

#### Highlights

- Support QDTrack algorithm in MOT ([#433](https://github.com/open-mmlab/mmtracking/pull/433)), ([#451](https://github.com/open-mmlab/mmtracking/pull/451)), ([#461](https://github.com/open-mmlab/mmtracking/pull/461)), ([#469](https://github.com/open-mmlab/mmtracking/pull/469))

#### Bug Fixes

- Support empty tensor for selsa aggregator ([#463](https://github.com/open-mmlab/mmtracking/pull/463))

### v0.11.0 (04/03/2022)

#### Highlights

- Support STARK algorithm in SOT ([#443](https://github.com/open-mmlab/mmtracking/pull/443)), ([#440](https://github.com/open-mmlab/mmtracking/pull/440)), ([#434](https://github.com/open-mmlab/mmtracking/pull/434)), ([#438](https://github.com/open-mmlab/mmtracking/pull/438)), ([#435](https://github.com/open-mmlab/mmtracking/pull/435)), ([#426](https://github.com/open-mmlab/mmtracking/pull/426))

- Support HOTA evaluation metrics for MOT ([#417](https://github.com/open-mmlab/mmtracking/pull/417))

#### New Features

- Support TAO dataset in MOT ([#415](https://github.com/open-mmlab/mmtracking/pull/415))

### v0.10.0 (10/02/2022)

#### New Features

- Support CPU training ([#404](https://github.com/open-mmlab/mmtracking/pull/404))

#### Improvements

- Refactor SOT datasets ([#401](https://github.com/open-mmlab/mmtracking/pull/401)), ([#402](https://github.com/open-mmlab/mmtracking/pull/402)), ([#393](https://github.com/open-mmlab/mmtracking/pull/393))

### v0.9.0 (05/01/2022)

#### Highlights

- Support arXiv 2021 manuscript 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box' ([#385](https://github.com/open-mmlab/mmtracking/pull/385)), ([#383](https://github.com/open-mmlab/mmtracking/pull/383)), ([#372](https://github.com/open-mmlab/mmtracking/pull/372))
- Support ICCV 2019 paper 'Video Instance Segmentation' ([#304](https://github.com/open-mmlab/mmtracking/pull/304)), ([#303](https://github.com/open-mmlab/mmtracking/pull/303)), ([#298](https://github.com/open-mmlab/mmtracking/pull/298)), ([#292](https://github.com/open-mmlab/mmtracking/pull/292))

#### New Features

- Support CrowdHuman dataset for MOT ([#366](https://github.com/open-mmlab/mmtracking/pull/366))
- Support VOT2018 dataset for SOT ([#305](https://github.com/open-mmlab/mmtracking/pull/305))
- Support YouTube-VIS dataset for VIS ([#290](https://github.com/open-mmlab/mmtracking/pull/290))

#### Bug Fixes

- Fix two significant bugs in SOT and provide new SOT pretrained models ([#349](https://github.com/open-mmlab/mmtracking/pull/349))

#### Improvements

- Refactor LaSOT, TrackingNet dataset and support GOT-10K datasets ([#296](https://github.com/open-mmlab/mmtracking/pull/296))
- Support persisitent workers ([#348](https://github.com/open-mmlab/mmtracking/pull/348))

### v0.8.0 (03/10/2021)

#### New Features

- Support OTB100 dataset in SOT ([#271](https://github.com/open-mmlab/mmtracking/pull/271))
- Support TrackingNet dataset in SOT ([#268](https://github.com/open-mmlab/mmtracking/pull/268))
- Support UAV123 dataset in SOT ([#260](https://github.com/open-mmlab/mmtracking/pull/260))

#### Bug Fixes

- Fix a bug in mot_param_search.py ([#270](https://github.com/open-mmlab/mmtracking/pull/270))

#### Improvements

- Use PyTorch sphinx theme ([#274](https://github.com/open-mmlab/mmtracking/pull/274))
- Use pycocotools instead of mmpycocotools ([#263](https://github.com/open-mmlab/mmtracking/pull/263))

### v0.7.0 (03/09/2021)

#### Highlights

- Release code of AAAI 2021 paper 'Temporal ROI Align for Video Object Recognition' ([#247](https://github.com/open-mmlab/mmtracking/pull/247))
- Refactor English documentations ([#243](https://github.com/open-mmlab/mmtracking/pull/243))
- Add Chinese documentations ([#248](https://github.com/open-mmlab/mmtracking/pull/248)), ([#250](https://github.com/open-mmlab/mmtracking/pull/250))

#### New Features

- Support fp16 training and testing ([#230](https://github.com/open-mmlab/mmtracking/pull/230))
- Release model using ResNeXt-101 as backbone for all VID methods ([#254](https://github.com/open-mmlab/mmtracking/pull/254))
- Support the results of Tracktor on MOT15, MOT16 and MOT20 datasets ([#217](https://github.com/open-mmlab/mmtracking/pull/217))
- Support visualization for single gpu test ([#216](https://github.com/open-mmlab/mmtracking/pull/216))

#### Bug Fixes

- Fix a bug in MOTP evaluation ([#235](https://github.com/open-mmlab/mmtracking/pull/235))
- Fix two bugs in reid training and testing ([#249](https://github.com/open-mmlab/mmtracking/pull/249))

#### Improvements

- Refactor anchor in SiameseRPN++ ([#229](https://github.com/open-mmlab/mmtracking/pull/229))
- Unify model initialization ([#235](https://github.com/open-mmlab/mmtracking/pull/235))
- Refactor unittest ([#231](https://github.com/open-mmlab/mmtracking/pull/231))

### v0.6.0 (30/07/2021)

#### Highlights

- Fix training bugs of all three tasks ([#219](https://github.com/open-mmlab/mmtracking/pull/219)), ([#221](https://github.com/open-mmlab/mmtracking/pull/221))

#### New Features

- Support error visualization for mot task ([#212](https://github.com/open-mmlab/mmtracking/pull/212))

#### Bug Fixes

- Fix a bug in SOT demo ([#213](https://github.com/open-mmlab/mmtracking/pull/213))

#### Improvements

- Use MMCV registry ([#220](https://github.com/open-mmlab/mmtracking/pull/220))
- Add README.md for reid training ([#210](https://github.com/open-mmlab/mmtracking/pull/210))
- Modify dict keys of the outputs of SOT ([#223](https://github.com/open-mmlab/mmtracking/pull/223))
- Add Chinese docs including install.md, quick_run.md, model_zoo.md, dataset.md ([#205](https://github.com/open-mmlab/mmtracking/pull/205)), ([#214](https://github.com/open-mmlab/mmtracking/pull/214))

### v0.5.3 (01/07/2021)

#### New Features

- Support ReID training ([#177](https://github.com/open-mmlab/mmtracking/pull/177)), ([#179](https://github.com/open-mmlab/mmtracking/pull/179)), ([#180](https://github.com/open-mmlab/mmtracking/pull/180)), ([#181](https://github.com/open-mmlab/mmtracking/pull/181)),
- Support MIM ([#158](https://github.com/open-mmlab/mmtracking/pull/158))

#### Bug Fixes

- Fix evaluation hook ([#176](https://github.com/open-mmlab/mmtracking/pull/176))
- Fix a typo in vid config ([#171](https://github.com/open-mmlab/mmtracking/pull/171))

#### Improvements

- Refactor nms config ([#167](https://github.com/open-mmlab/mmtracking/pull/167))

### v0.5.2 (03/06/2021)

#### Improvements

- Fixed typos ([#104](https://github.com/open-mmlab/mmtracking/commit/3ccc9b79ce6e14e013268d0dbb53462c0432f357), [#121](https://github.com/open-mmlab/mmtracking/commit/fadcd811df095781fbbdc7c47f8dac1305555461), [#145](https://github.com/open-mmlab/mmtracking/commit/48a47868abd9a0d96c010fc3f85cba1bd2854a9b))
- Added conference reference ([#111](https://github.com/open-mmlab/mmtracking/commit/9a3c463b087cdee201a9345f270f6c01e116cf2c))
- Updated the link of CONTRIBUTING to mmcv ([#112](https://github.com/open-mmlab/mmtracking/commit/b725e63463b1bd795fd3c3000b30ef37832a844d))
- Adapt updates in mmcv (FP16Hook) ([#114](https://github.com/open-mmlab/mmtracking/commit/49f910878345250d22fd5da1104f1fb227244939), [#119](https://github.com/open-mmlab/mmtracking/commit/f1df53dd8e571f4674867919d1886b9fb2024bf9))
- Added bibtex and links to other codebases ([#122](https://github.com/open-mmlab/mmtracking/commit/1b456423e0aeddb52e7c29e5b0ec3d48e058c615))
- Added docker files ([#124](https://github.com/open-mmlab/mmtracking/commit/a01c3e8fff97a2b8eebc8d28e3e9d9a360ffbc3c))
- Used `collect_env` in mmcv ([#129](https://github.com/open-mmlab/mmtracking/commit/0055947c4d19c8921c32ce128ae0314d61e593d2))
- Added and updated Chinese README ([#135](https://github.com/open-mmlab/mmtracking/commit/ecc83b5e6523582b92196095eb21d72d654322f2), [#147](https://github.com/open-mmlab/mmtracking/commit/19004b6eeca594a2179d8b3a3622764e1753aa4d), [#148](https://github.com/open-mmlab/mmtracking/commit/dc367868453fdcb528041176a59ede368f0e2053))

### v0.5.1 (01/02/2021)

#### Bug Fixes

- Fixed ReID checkpoint loading ([#80](https://github.com/open-mmlab/mmtracking/pull/80))
- Fixed empty tensor in `track_result` ([#86](https://github.com/open-mmlab/mmtracking/pull/86))
- Fixed `wait_time` in MOT demo script ([#92](https://github.com/open-mmlab/mmtracking/pull/92))

#### Improvements

- Support single-stage detector for DeepSORT ([#100](https://github.com/open-mmlab/mmtracking/pull/100))

### v0.5.0 (04/01/2021)

#### Highlights

- MMTracking is released!

#### New Features

- Support video object detection methods: [DFF](https://arxiv.org/abs/1611.07715), [FGFA](https://arxiv.org/abs/1703.10025), [SELSA](https://arxiv.org/abs/1907.06390)
- Support multi object tracking methods: [SORT](https://arxiv.org/abs/1602.00763)/[DeepSORT](https://arxiv.org/abs/1703.07402), [Tracktor](https://arxiv.org/abs/1903.05625)
- Support single object tracking methods: [SiameseRPN++](https://arxiv.org/abs/1812.11703)
