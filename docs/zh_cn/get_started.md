## 依赖

- Linux | macOS | Windows
- Python 3.6+
- PyTorch 1.6+
- CUDA 9.2+ （如果基于 PyTorch 源码安装，也能够支持 CUDA 9.0）
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

MMTracking, MMEngine, MMCV 和 MMDetection 的兼容版本如下。请安装正确的版本以避免安装出现问题。

| MMTracking version | MMEngine version |      MMCV version      |   MMDetection version   |
| :----------------: | :--------------: | :--------------------: | :---------------------: |
|        1.x         | mmengine>=0.1.0  | mmcv>=2.0.0rc1,\<2.0.0 | mmdet>=3.0.0rc0,\<3.0.0 |

## 安装

### 详细说明

1. 创建并激活 conda 虚拟环境

   ```shell
   conda create -n open-mmlab python=3.9 -y
   conda activate open-mmlab
   ```

2. 基于 [PyTorch 官方说明](https://pytorch.org/) 安装 PyTorch 和 torchvision。这里我们使用 PyTorch 1.10.0 和 CUDA 11.1。
   您也可以指定版本号切换其他版本。

   **使用 conda 安装**

   ```shell
   conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
   ```

   **使用 pip 安装**

   ```shell
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

3. 安装 MMEngine

   ```shell
   pip install mmengine
   ```

4. 安装 mmcv， 我们建议使用预编译包来安装：

   ```shell
   pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   ```

   mmcv 仅在 PyTorch 1.x.0 上编译，因为通常 1.x.0 版本与 1.x.1 版本具有兼容性。如果您的 PyTorch 版本是 1.x.1，则可以安装使用 PyTorch 1.x.0 编译的 mmcv，并且通常运行情况良好。

   ```shell
   # 我们可以忽略 Pytorch 的微型版本
   pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
   ```

   请参阅 [此处](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 的不同版本的 MMCV 与不同的 Pytorch 和 CUDA 版本兼容。
   您可以选择通过以下命令选择从源中编译 MMCV。

   ```shell
   git clone -b 2.x https://github.com/open-mmlab/mmcv.git
   cd mmcv
   MMCV_WITH_OPS=1 pip install -e .  # 包含 CUDA 选项的软件包MMCV将在此步骤之后安装
   # pip install -e .  # 不包含 CUDA 选项的软件包 MMCV 将在此步骤之后安装
   cd ..
   ```

   **重要提示**:  如果安装了 mmcv，则需要首先运行 pip 卸载 mmcv-lite。因为如果 mmcv-lite 和 mmcv 都已安装，会出现 `ModuleNotFoundError`。

5. 安装 MMDetection

   ```shell
   pip install 'mmdet>=3.0.0rc0'
   ```

   如果您想修改代码，您也可以选择从源代码编译 MMDetection:

   ```shell
   git clone -b 3.x https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -r requirements/build.txt
   pip install -v -e .  # 或者 "python setup.py develop"
   ```

6. 克隆 MMTracking 存储库。

   ```shell
   git clone -b 1.x https://github.com/open-mmlab/mmtracking.git
   cd mmtracking
   ```

7. 安装相关依赖，然后安装 MMTracking。

   ```shell
   pip install -r requirements/build.txt
   pip install -v -e .  # 或者 "python setup.py develop"
   ```

8. 安装额外的依赖

- 用于 MOT 数据集评估（可选）：

  ```shell
  pip install git+https://github.com/JonathonLuiten/TrackEval.git
  ```

- 用于 VOT 数据集评估（可选）：

  ```shell
  pip install git+https://github.com/votchallenge/toolkit.git
  ```

- 用于 LVIS 数据集评估（可选）：

  ```shell
  pip install git+https://github.com/lvis-dataset/lvis-api.git
  ```

- 用于 TAO 数据集评估（可选）：

  ```shell
  pip install git+https://github.com/TAO-Dataset/tao.git
  ```

注意：

a. 根据上面的说明，MMTracking 安装在 `dev` 模式下，对代码的任何本地修改都将生效，无需重新安装。

b. 如果您想用 `opencv-python-headless` 替换 `opencv-python`， 您可以在安装 MMCV 之前安装它。

### 从头开始的安装脚本

假设您已经安装了 CUDA 10.1，下面是一个使用 conda 安装 MMTracking 的完整脚本。

```shell
conda create -n open-mmlab python=3.9 -y
conda activate open-mmlab

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

pip install mmengine

# 安装最新的 mmcv
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# 安装 mmdetection
pip install 'mmdet>=3.0.0rc0'

# 安装 mmtracking
git clone -b 1.x https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install git+https://github.com/votchallenge/toolkit.git (optional)
pip install git+https://github.com/lvis-dataset/lvis-api.git (optional)
pip install git+https://github.com/TAO-Dataset/tao.git (optional)
```

### 使用多个 MMTracking 版本进行开发

训练和测试脚本已经修改了 `PYTHONPATH`，以确保脚本使用当前目录中的 MMTracking。

要使用安装在环境中默认的 MMTracking 而不是您正在使用的，可以删除出现在相关脚本中的以下代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## 验证

为了验证 MMTracking 和所需环境是否正确安装, 我们可以运行 MOT, VIS, VID and SOT [演示脚本](https://github.com/open-mmlab/mmtracking/blob/1.x/demo/) 中的**任意一个** 。

以下是 MOT 演示的示例：

```shell
python demo/demo_mot_vis.py \
    configs/mot/deepsort/deepsort_faster-rcnn-r50-fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4
```

如果您想运行更多其他演示，您可以参考[推理指南](./user_guides/3_inference.md)
