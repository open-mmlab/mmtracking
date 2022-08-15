# Benchmark and Model Zoo

## Common settings

- We use distributed training.

- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.

- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.

- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time. Results are obtained with the script `tools/analysis/benchmark.py` which computes the average time on 2000 images.

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

Please refer to [DFF](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/dff) for details.

### FGFA (ICCV 2017)

Please refer to [FGFA](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/fgfa) for details.

### SELSA (ICCV 2019)

Please refer to [SELSA](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/selsa) for details.

### Temporal RoI Align (AAAI 2021)

Please refer to [Temporal RoI Align](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/temporal_roi_align) for details.

## Baselines of multiple object tracking

### SORT/DeepSORT (ICIP 2016/2017)

Please refer to [SORT/DeepSORT](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/deepsort) for details.

### Tracktor (ICCV 2019)

Please refer to [Tracktor](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/tracktor) for details.

### QDTrack (CVPR 2021)

Please refer to [QDTrack](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/qdtrack) for details.

### ByteTrack (arXiv 2021)

Please refer to [ByteTrack](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/bytetrack) for details.

## Baselines of single object tracking

### SiameseRPN++ (CVPR 2019)

Please refer to [SiameseRPN++](https://github.com/open-mmlab/mmtracking/blob/master/configs/sot/siamese_rpn) for details.

### STARK (ICCV 2021)

Please refer to [STARK](https://github.com/open-mmlab/mmtracking/blob/master/configs/sot/stark) for details.

## Baselines of video instance segmentation

### MaskTrack R-CNN (ICCV 2019)

Please refer to [MaskTrack R-CNN](https://github.com/open-mmlab/mmtracking/blob/master/configs/vis/masktrack_rcnn) for details.
