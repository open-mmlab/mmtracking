## Prerequisites

- Linux | macOS | Windows
- Python 3.6+
- PyTorch 1.6+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

The compatible MMTracking, MMEngine, MMCV, and MMDetection versions are as below. Please install the correct version to avoid installation issues.

| MMTracking version | MMEngine version |      MMCV version      |   MMDetection version   |
| :----------------: | :--------------: | :--------------------: | :---------------------: |
|        1.x         | mmengine>=0.1.0  | mmcv>=2.0.0rc1,\<2.0.0 | mmdet>=3.0.0rc0,\<3.0.0 |
|      1.0.0rc1      | mmengine>=0.1.0  | mmcv>=2.0.0rc1,\<2.0.0 | mmdet>=3.0.0rc0,\<3.0.0 |

## Installation

### Detailed Instructions

1. Create a conda virtual environment and activate it.

   ```shell
   conda create -n open-mmlab python=3.9 -y
   conda activate open-mmlab
   ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/). Here we use PyTorch 1.10.0 and CUDA 11.1.
   You may also switch to other version by specifying the version number.

   **Install with conda**

   ```shell
   conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
   ```

   **Install with pip**

   ```shell
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

3. Install MMEngine

   ```shell
   pip install mmengine
   ```

4. Install mmcv, we recommend you to install the pre-build package as below.

   ```shell
   pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   ```

   mmcv is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv compiled with PyTorch 1.x.0 and it usually works well.

   ```shell
   # We can ignore the micro version of PyTorch
   pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
   ```

   See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
   Optionally you can choose to compile mmcv from source by the following command

   ```shell
   git clone -b 2.x https://github.com/open-mmlab/mmcv.git
   cd mmcv
   MMCV_WITH_OPS=1 pip install -e .  # package mmcv, which contains cuda ops, will be installed after this step
   # pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
   cd ..
   ```

   **Important**: You need to run pip uninstall mmcv-lite first if you have mmcv installed. Because if mmcv-lite and mmcv are both installed, there will be ModuleNotFoundError.

5. Install MMDetection

   ```shell
   pip install 'mmdet>=3.0.0rc0'
   ```

   Optionally, you can also build MMDetection from source in case you want to modify the code:

   ```shell
   git clone -b 3.x https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -r requirements/build.txt
   pip install -v -e .  # or "python setup.py develop"
   ```

6. Clone the MMTracking repository.

   ```shell
   git clone -b 1.x https://github.com/open-mmlab/mmtracking.git
   cd mmtracking
   ```

7. Install build requirements and then install MMTracking.

   ```shell
   pip install -r requirements/build.txt
   pip install -v -e .  # or "python setup.py develop"
   ```

8. Install extra dependencies

- For MOT evaluation (required):

  ```shell
  pip install git+https://github.com/JonathonLuiten/TrackEval.git
  ```

- For VOT evaluation (optional)

  ```shell
  pip install git+https://github.com/votchallenge/toolkit.git
  ```

- For LVIS evaluation (optional):

  ```shell
  pip install git+https://github.com/lvis-dataset/lvis-api.git
  ```

- For TAO evaluation (optional):

  ```shell
  pip install git+https://github.com/TAO-Dataset/tao.git
  ```

Note:

a. Following the above instructions, MMTracking is installed on `dev` mode
, any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMTracking with conda.

```shell
conda create -n open-mmlab python=3.9 -y
conda activate open-mmlab

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

pip install mmengine

# install the latest mmcv
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install mmdetection
pip install 'mmdet>=3.0.0rc0'

# install mmtracking
git clone -b 1.x https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install git+https://github.com/votchallenge/toolkit.git (optional)
pip install git+https://github.com/lvis-dataset/lvis-api.git (optional)
pip install git+https://github.com/TAO-Dataset/tao.git (optional)
```

### Developing with multiple MMTracking versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMTracking in the current directory.

To use the default MMTracking installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMTracking and the required environment are installed correctly, we can run **one of** MOT, VIS, VID and SOT [demo scripts](https://github.com/open-mmlab/mmtracking/blob/1.x/demo/):

Here is an example for MOT demo:

```shell
python demo/demo_mot_vis.py \
    configs/mot/deepsort/deepsort_faster-rcnn-r50-fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4
```

If you want to run more other demos, you can refer to [inference guides](./user_guides/3_inference.md)
