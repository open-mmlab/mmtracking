## Prerequisites

- Linux | macOS | Windows
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html)
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

3. Install mmcv-full, we recommend you to install the pre-build package as below.

   ```shell
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   ```

   mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.


   ```shell
   # We can ignore the micro version of PyTorch
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
   ```

   See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
   Optionally you can choose to compile mmcv from source by the following command

   ```shell
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
   # pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
   cd ..
   ```

   **Important**: You need to run pip uninstall mmcv first if you have mmcv installed. Because if mmcv and mmcv-full are both installed, there will be ModuleNotFoundError.

4. Install MMEngine

   ```shell
   pip install mmengine
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

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install mmdetection
pip install mmdet

# install mmtracking
git clone https://github.com/open-mmlab/mmtracking.git
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

To verify whether MMTracking and the required environment are installed correctly, we can run MOT, VIS, VID and SOT [demo scripts](https://github.com/open-mmlab/mmtracking/tree/master/demo/):

```
python demo/{DEMO_FILE} \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --checkpoint ${CHECKPOINT_FILE} \
    [--output ${OUTPUT}] \
    [--device ${DEVICE}] \
    [--show] \
    [--fps ${FPS]
```
Required arguments：

- `CONFIG_FILE`: the path of configuration file.
- `INPUT`: a given video or a folder that contains continuous images.

    **Note** that if you use a folder as the input,
    + the image names there must be **sortable**, which means we can re-order the images according to the numbers contained in the filenames.
    + we now only support reading the images whose filenames end with '.jpg', '.jpeg' and '.png'.
    + the argument `--fps` must be specified in the command script if you want to show or save the tracking video.

Optional arguments:

- `CHECKPOINT_FILE`: The checkpoint is optional in case that you already set up the pretrained models in the config by the key `pretrains`.
- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.
- `FPS`: The fps of the video. Only used for show or saving the video.

For VID/MOT/VIS tasks, you can also specify the argument `--score-thr` in the command script to set the threshold of score to filter bboxes.

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`. We provide some examples for different task:

#### Example of VID

```shell
python ./demo/demo_vid.py \
    ./configs/vid/selsa/selsa_faster-rcnn-resnet50-dc5_8x1bs-7e_imagenetvid.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth \
    --output ${OUTPUT} \
    --show
```

#### Example of MOT

```shell
python demo/demo_mot_vis.py \
    configs/mot/deepsort/deepsort_faster-rcnn-resnet50-fpn_8x2bs-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4 \
```

**Note**:
+ Deepsort don't need to load checkpoint since the tracker don't have to be trained. However, for other MOT methods, such as byteytrack and qdtrack, you need to add an input argument `--checkpoint` in the commend script.
+ When running `demo_mot_vis.py`, we suggest you use the config containing `private`, since `private` means the MOT method doesn't need external detections.

#### Example of VIS

```shell
python demo/demo_mot_vis.py \
    configs/vis/masktrack_rcnn/masktrack-rcnn_mask-rcnn-resnet50-fpn_8x1bs-12e_youtubevis2019.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth \
    --output ${OUTPUT} \
    --show
```

#### Example of SOT

```shell
python ./demo/demo_sot.py \
    ./configs/sot/siamese_rpn/siamese-rpn_resnet50_8x28bs-20e_imagenetvid-imagenetdet-coco_test-lasot.py \
    --input ${VIDEO_FILE} \
    --checkpoint checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth \
    --output ${OUTPUT} \
    --show
```
**Note** that the SOT demo support load the annotation of the first frame from the annotation file. You can specify an argument 
`--gt_bbox_file` which menas the gt_bbox file path of the video. The first line annotatioin in the file will be used to initilize the tracker. If not specified, you would draw init bbox of the video manually.**


