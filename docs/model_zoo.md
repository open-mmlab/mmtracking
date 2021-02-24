# Benchmark and Model Zoo

## Common settings

- We use distributed training.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time. Results are obtained with the script `tools/benchmark.py` which computes the average time on 2000 images.
- Speed benchmark environments

    HardWare
    - 8 NVIDIA Tesla V100 (32G) GPUs
    - Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

    Software environment
    - Python 3.7
    - PyTorch 1.5
    - CUDA 10.1
    - CUDNN 7.6.03
    - NCCL 2.4.08

## Baselines of video object detection

### DFF (CVPR 2017)

Please refer to [DFF](../configs/vid/dff/README.md) for details.

### FGFA (ICCV 2017)

Please refer to [FGFA](../configs/vid/fgfa/README.md) for details.

### SELSA (ICCV 2019)

Please refer to [SELSA](../configs/vid/selsa/README.md) for details.

## Baselines of multiple object tracking

### SORT/DeepSORT (ICIP 2016/2017)

Please refer to [SORT/DeepSORT](../configs/mot/deepsort/README.md) for details.

### Tracktor (ICCV 2019)

Please refer to [Tracktor](../configs/mot/tracktor/README.md) for details.

## Baselines of single object tracking

### SiameseRPN++ (CVPR 2019)

Please refer to [SiameseRPN++](../configs/sot/siamese_rpn/README.md) for details.
