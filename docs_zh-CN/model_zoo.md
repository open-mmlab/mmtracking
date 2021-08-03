# 基准测试与模型库

## 通用设置

- 我们默认使用分布式训练。
- 所有 pytorch 风格的预训练骨干网络都是来自 Pytorch 的模型库。
- 为了与其他代码库进行公平比较，我们以全部 8 个 GPU 的 `torch.cuda.max_memory_allocated()` 的最大值作为 GPU 显存使用量。请注意，此值通常小于 `nvidia-smi` 显示的值。
- 该推理时间不包含数据加载时间，推理时间结果是通过脚本 `tools/benchmark.py` 获得的，该脚本计算处理 2000 张图像的平均时间。
- 速度基准测试的环境如下：

    硬件环境：
    - 8 NVIDIA Tesla V100 (32G) GPUs
    - Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

    软件环境：
    - Python 3.7
    - PyTorch 1.5
    - CUDA 10.1
    - CUDNN 7.6.03
    - NCCL 2.4.08

## 视频目标检测基线

### DFF (CVPR 2017)

详情请参考 [DFF](../configs/vid/dff/README.md)。

### FGFA (ICCV 2017)

详情请参考 [FGFA](../configs/vid/fgfa/README.md)。

### SELSA (ICCV 2019)

详情请参考 [SELSA](../configs/vid/selsa/README.md)。

## 多目标跟踪基线

### SORT/DeepSORT (ICIP 2016/2017)

详情请参考 [SORT/DeepSORT](../configs/mot/deepsort/README.md)。

### Tracktor (ICCV 2019)

详情请参考 [Tracktor](../configs/mot/tracktor/README.md)。

## 单目标跟踪基线

### SiameseRPN++ (CVPR 2019)

详情请参考 [SiameseRPN++](../configs/sot/siamese_rpn/README.md)。
