## 依赖

- Linux or macOS
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (如果您从源代码构建 PyTorch, 那么 CUDA9.0也是兼容的)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation)

兼容的 `MMTracking`, `MMCV` 和 `MMDetection` 版本如下，请安装正确的版本以避免安装问题。

|  MMTracking version |       MMCV version       |      MMDetection version      |
|:-------------------:|:------------------------:|:-----------------------------:|
|        master       | mmcv-full>=1.3.8, <1.4.0 |       MMDetection>=2.14.0      |
|        0.5.3        | mmcv-full>=1.3.8, <1.4.0 |       MMDetection>=2.14.0     |
|        0.5.2        | mmcv-full>=1.3.3, <1.4.0 |       MMDetection=2.12.0      |

## 安装

### 详细说明

1. 创建一个`conda`虚拟环境并激活它。

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. 按照[官方说明](https://pytorch.org/)安装`PyTorch`和`torchvision` , 例如

    ```shell
    conda install pytorch torchvision -c pytorch
    ```
    注意：请确保您编译的`CUDA`版本和运行时`CUDA`版本匹配，您可以在[PyTorch官方](https://pytorch.org/)查看支持`CUDA`版本的预编译包。

    `E.g.1` 如果您在 `/usr/local/cuda` 安装了`CUDA 10.1` 并且想要安装`PyTorch 1.5`，您需要安装预先构建的带有`CUDA 10.1`的PyTorch版本。

       ```shell
    conda install pytorch==1.5 cudatoolkit=10.1 torchvision -c pytorch
       ```
    `E.g.2` 如果您在 `/usr/local/cuda` 安装了`CUDA 9.2` 并且想要安装`PyTorch 1.3.1`，您需要安装预先构建的带有`CUDA 9.2`的PyTorch版本。

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```
    如果您从源代码构建`PyTorch`而不是直接安装预构建包，您能够使用更多的CUDA版本例如`9.0`。

3. 安装`mmcv-full`，我们推荐您安装以下预构建包：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
    ```

    请参阅[这里](https://github.com/open-mmlab/mmcv#install-with-pip)了解不同版本的`MMCV`与不同版本的`PyTorch`和`CUDA`的兼容情况。或者您可以使用一下命令从源码编译`mmcv`。

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    或者直接运行

    ```shell
    pip install mmcv-full
    ```
    
4. 安装`MMDetection`。

    ```shell
    pip install mmdet
    ```

    如果您想修改代码，您也可以从源码构建`MMDetection`：

    ```shell
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```
    
5. 克隆 `MMTracking` 存储库。

    ```shell
    git clone https://github.com/open-mmlab/mmtracking.git
    cd mmtracking
    ```
    
6. 安装`requirements`然后安装`MMTracking`。

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

注意：

1. 按照上述说明，`MMTracking`以`dev`模式安装，对本地代码所做的任何修改无需重新安装即可生效。
2. 如果您想使用`opencv-python-headless`代替`opencv-python`，您可以在安装`MMCV`之后安装它。

### 从头设置的脚本

假如您已经安装了`CUDA 10.1`，以下为使用`conda`设置`MMTracking`的完整脚本。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmdetection
pip install mmdet

# install mmtracking
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
```
### 使用多个 `MMTracking` 版本进行开发

训练和测试脚本已经修改了`PYTHONPATH`以确保脚本使用当前目录的`MMTracking`。

要使用环境中默认安装的`MMTracking`而不是您正在使用的版本，您可以删除脚本中的以下行:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

### 验证

为了验证`MMTracking`和所需环境是否安装正确，我们可以运行`MOT`、`VID`、`SOT`的演示脚本。

运行`MOT`演示脚本您可以看到输出一个命名为`mot.mp4`的视频文件：

```shell
python demo/demo_mot.py configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py --input demo/demo.mp4 --output mot.mp4
```

