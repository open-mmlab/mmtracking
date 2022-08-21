## 更新日志

### v0.6.0 (30/07/2021)

#### 亮点

- 修复三个任务的训练问题 ([#219](https://github.com/open-mmlab/mmtracking/pull/219)), ([#221](https://github.com/open-mmlab/mmtracking/pull/221))

#### 新特性

- 支持多目标跟踪任务的错误分析可视化 ([#212](https://github.com/open-mmlab/mmtracking/pull/212))

#### 问题修复

- 修复单目标跟踪演示中的一个问题 ([#213](https://github.com/open-mmlab/mmtracking/pull/213))

#### 提升

- 使用 MMCV 中的注册器 ([#220](https://github.com/open-mmlab/mmtracking/pull/220))
- 增加重识别任务训练教程 ([#210](https://github.com/open-mmlab/mmtracking/pull/210))
- 修改单目标跟踪输出字典的键值 ([#223](https://github.com/open-mmlab/mmtracking/pull/223))
- 增加四篇中文文档 install.md, quick_run.md, model_zoo.md, dataset.md ([#205](https://github.com/open-mmlab/mmtracking/pull/205)), ([#214](https://github.com/open-mmlab/mmtracking/pull/214))

### v0.5.3 (01/07/2021)

#### 新特性

- 支持重识别任务的训练 ([#177](https://github.com/open-mmlab/mmtracking/pull/177), [#179](https://github.com/open-mmlab/mmtracking/pull/179), [#180](https://github.com/open-mmlab/mmtracking/pull/180), [#181](https://github.com/open-mmlab/mmtracking/pull/181))
- 支持 MIM ([#158](https://github.com/open-mmlab/mmtracking/pull/158))

#### 问题修复

- 修复评测钩子 ([#176](https://github.com/open-mmlab/mmtracking/pull/176))
- 修复视频目标检测配置中的错字 ([#171](https://github.com/open-mmlab/mmtracking/pull/171))

#### 提升

- 重构 nms 配置 ([#167](https://github.com/open-mmlab/mmtracking/pull/167))

### v0.5.2 (03/06/2021)

#### 提升

- 修复错字 ([#104](https://github.com/open-mmlab/mmtracking/commit/3ccc9b79ce6e14e013268d0dbb53462c0432f357), [#121](https://github.com/open-mmlab/mmtracking/commit/fadcd811df095781fbbdc7c47f8dac1305555461), [#145](https://github.com/open-mmlab/mmtracking/commit/48a47868abd9a0d96c010fc3f85cba1bd2854a9b))
- 增加会议引用 ([#111](https://github.com/open-mmlab/mmtracking/commit/9a3c463b087cdee201a9345f270f6c01e116cf2c))
- 更新 CONTRIBUTING 链接到 mmcv ([#112](https://github.com/open-mmlab/mmtracking/commit/b725e63463b1bd795fd3c3000b30ef37832a844d))
- 调整 mmcv 中的更新 (FP16Hook) ([#114](https://github.com/open-mmlab/mmtracking/commit/49f910878345250d22fd5da1104f1fb227244939), [#119](https://github.com/open-mmlab/mmtracking/commit/f1df53dd8e571f4674867919d1886b9fb2024bf9))
- 添加了指向其他代码库的 bibtex 和链接 ([#122](https://github.com/open-mmlab/mmtracking/commit/1b456423e0aeddb52e7c29e5b0ec3d48e058c615))
- 添加 docker 文件 ([#124](https://github.com/open-mmlab/mmtracking/commit/a01c3e8fff97a2b8eebc8d28e3e9d9a360ffbc3c))
- 使用 mmcv 中的 `collect_env` ([#129](https://github.com/open-mmlab/mmtracking/commit/0055947c4d19c8921c32ce128ae0314d61e593d2))
- 增加和更新中文教程 ([#135](https://github.com/open-mmlab/mmtracking/commit/ecc83b5e6523582b92196095eb21d72d654322f2), [#147](https://github.com/open-mmlab/mmtracking/commit/19004b6eeca594a2179d8b3a3622764e1753aa4d), [#148](https://github.com/open-mmlab/mmtracking/commit/dc367868453fdcb528041176a59ede368f0e2053))

### v0.5.1 (01/02/2021)

#### 问题修复

- 修复重识别模型权重文件的导入 ([#80](https://github.com/open-mmlab/mmtracking/pull/80))
- 修复 `track_result` 中的空张量 ([#86](https://github.com/open-mmlab/mmtracking/pull/86))
- 修复多目标跟踪演示脚本中的 `wait_time` ([#92](https://github.com/open-mmlab/mmtracking/pull/92))

#### 提升

- 支持 DeepSORT 使用单阶段检测器 ([#100](https://github.com/open-mmlab/mmtracking/pull/100))

### v0.5.0 (04/01/2021)

#### 亮点

- MMTracking 已经发布!

#### 新特性

- 支持的视频目标检测方法: [DFF](https://arxiv.org/abs/1611.07715), [FGFA](https://arxiv.org/abs/1703.10025), [SELSA](https://arxiv.org/abs/1907.06390)
- 支持的多目标跟踪方法: [SORT](https://arxiv.org/abs/1602.00763)/[DeepSORT](https://arxiv.org/abs/1703.07402), [Tracktor](https://arxiv.org/abs/1903.05625)
- 支持的单目标跟踪方法: [SiameseRPN++](https://arxiv.org/abs/1812.11703)
