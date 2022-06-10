## Prerequisites

- Linux | macOS | Windows
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

The compatible MMTracking, MMCV, and MMDetection versions are as below. Please install the correct version to avoid installation issues.

| MMTracking version |        MMCV version        | MMDetection version |
| :----------------: | :------------------------: | :-----------------: |
|       master       | mmcv-full>=1.3.17, \<1.6.0 | MMDetection>=2.19.1 |
|       0.13.0       | mmcv-full>=1.3.17, \<1.6.0 | MMDetection>=2.19.1 |
|       0.12.0       | mmcv-full>=1.3.17, \<1.5.0 | MMDetection>=2.19.1 |
|       0.11.0       | mmcv-full>=1.3.17, \<1.5.0 | MMDetection>=2.19.1 |
|       0.10.0       | mmcv-full>=1.3.17, \<1.5.0 | MMDetection>=2.19.1 |
|       0.9.0        | mmcv-full>=1.3.17, \<1.5.0 | MMDetection>=2.19.1 |
|       0.8.0        | mmcv-full>=1.3.8, \<1.4.0  | MMDetection>=2.14.0 |
|       0.7.0        | mmcv-full>=1.3.8, \<1.4.0  | MMDetection>=2.14.0 |
|       0.6.0        | mmcv-full>=1.3.8, \<1.4.0  | MMDetection>=2.14.0 |

## Installation

### Detailed Instructions

1. Create a conda virtual environment and activate it.

   ```shell
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   Note: Make sure that your compilation CUDA version and runtime CUDA version match.
   You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

   `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
   PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

   ```shell
   conda install pytorch==1.5 cudatoolkit=10.1 torchvision -c pytorch
   ```

   `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
   PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

   ```shell
   conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
   ```

   If you build PyTorch from source instead of installing the prebuilt package,
   you can use more CUDA versions such as 9.0.

3. Install extra dependencies for VOT evaluation (optional)

   If you need to evaluate on VOT Challenge, please install the vot-toolkit before the installation of mmcv and mmdetection to avoid possible numpy version requirement conflict among some dependencies.

   ```shell
   pip install git+https://github.com/votchallenge/toolkit.git

   ```

4. Install mmcv-full, we recommend you to install the pre-build package as below.

   ```shell
   # pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
   ```

   mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

   ```shell
   # We can ignore the micro version of PyTorch
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
   ```

   See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
   Optionally you can choose to compile mmcv from source by the following command

   ```shell
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
   cd ..
   ```

5. Install MMDetection

   ```shell
   pip install mmdet
   ```

   Optionally, you can also build MMDetection from source in case you want to modify the code:

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -r requirements/build.txt
   pip install -v -e .  # or "python setup.py develop"
   ```

6. Clone the MMTracking repository.

   ```shell
   git clone https://github.com/open-mmlab/mmtracking.git
   cd mmtracking
   ```

7. Install build requirements and then install MMTracking.

   ```shell
   pip install -r requirements/build.txt
   pip install -v -e .  # or "python setup.py develop"
   ```

8. Install extra dependencies

- For MOTChallenge evaluation:

  ```shell

  pip install git+https://github.com/JonathonLuiten/TrackEval.git
  ```

- For LVIS evaluation:

  ```shell
  pip install git+https://github.com/lvis-dataset/lvis-api.git
  ```

- For TAO evaluation:

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
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

pip install git+https://github.com/votchallenge/toolkit.git (optional)
# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmdetection
pip install mmdet

# install mmtracking
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/TAO-Dataset/tao.git
```

### Developing with multiple MMTracking versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMTracking in the current directory.

To use the default MMTracking installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMTracking and the required environment are installed correctly, we can run MOT, VID, SOT demo script.

For example, run MOT demo and you will see a output video named `mot.mp4`:

```shell
python demo/demo_mot_vis.py configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py --input demo/demo.mp4 --output mot.mp4
```
