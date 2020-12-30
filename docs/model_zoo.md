# Benchmark and Model Zoo

## Common settings

- We use distributed training.
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs.
Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time.
Results are obtained with the script `tools/benchmark.py` which computes the average time on 2000 images.

## Baselines of video object detection

### DFF
Please refer to [DFF](../configs/vid/dff/README.md) for details.

### FGFA
Please refer to [FGFA](../configs/vid/fgfa/README.md) for details.

### SELSA
Please refer to [SELSA](../configs/vid/selsa/README.md) for details.

## Baselines of multi object tracking

### SORT/DeepSORT
Please refer to [SORT/DeepSORT](../configs/mot/deepsort/README.md) for details.

### Tracktor
Please refer to [Tracktor](../configs/mot/tracktor/README.md) for details.

## Baselines of single object tracking

### SiameseRPN++
Please refer to [SiameseRPN++](../configs/sot/siamese_rpn/README.md) for details.

## Speed Benchmark

### HardWare
- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### Software environment
- Python 3.7
- PyTorch 1.5
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08

### Training speed and memory of video object detection

For fair comparison, we benchmark all implementations with ResNet-101.

The training speed is reported as followed, in terms of second per iter (s/iter). The lower, the better.

| Method       | speed      | memory      |
|--------------|------------|-------------|
| DFF          | 0.175      | 3333        |
| FGFA         | 0.310      | 5935        |
| SELSA        | 0.300      | 5305        |

### Inference speed of video object detection

For fair comparison, we benchmark all implementations with ResNet-101.

The inference speed is measured with fps (img/s) on a single GPU, the higher, the better.

| Method       | speed      |
|--------------|------------|
| DFF          | 39.8       |
| FGFA         | 6.4        |
| SELSA        | 7.2        |

### Training speed and memory of multi object tracking



### Inference speed of multi object tracking



### Training speed and memory of single object tracking



### Inference speed of single object tracking
