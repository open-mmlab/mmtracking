## Changelog

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
