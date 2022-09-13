# Mask2Former for Video Instance Segmentation

## Abstract

<!-- [ABSTRACT] -->

We find Mask2Former also achieves state-of-the-art performance on video instance segmentation without modifying the architecture, the loss or even the training pipeline. In this report, we show universal image segmentation architectures trivially generalize to video segmentation by directly predicting 3D segmentation volumes. Specifically, Mask2Former sets a new state-of-the-art of 60.4 AP on YouTubeVIS-2019 and 52.6 AP on YouTubeVIS-2021. We believe Mask2Former is also capable of handling video semantic and panoptic segmentation, given its versatility in image segmentation. We hope this will make state-of-theart video segmentation research more accessible and bring more attention to designing universal image and video segmentation architectures.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/46072190/188271377-164634a5-4d65-4161-8a69-2d0eaf2791f8.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

## Results and models of Mask2Former on YouTube-VIS 2021 validation dataset

Note: Codalab has closed the evaluation portal of `YouTube-VIS 2019`, so we do not provide the results of `YouTube-VIS 2019` at present. If you want to evaluate the results of `YouTube-VIS 2021`, at present, you can submit the result to the evaluation portal of `YouTube-VIS 2022`. The value of `AP_S` is the result of `YouTube-VIS 2021`.

|          Method          | Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  |                                 Config                                  |                                                                                                                                                    Download                                                                                                                                                    |
| :----------------------: | :------: | :-----: | :-----: | :------: | :------------: | :--: | :---------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       Mask2Former        |   R-50   | pytorch |   8e    |   6.0    |       -        | 41.2 |           [config](mask2former_r50_8xb2-8e_youtubevis2021.py)           |                     [model](https://download.openmmlab.com/mmtracking/vis/mask2former/mask2former_r50_8xb2-8e_youtubevis2021_20220818_164043-1cab1219.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/mask2former/mask2former_r50_8xb2-8e_youtubevis2021_20220818_164043.json)                     |
|       Mask2Former        |  R-101   | pytorch |   8e    |   7.5    |       -        | 42.3 |          [config](mask2former_r101_8xb2-8e_youtubevis2021.py)           |                    [model](https://download.openmmlab.com/mmtracking/vis/mask2former/mask2former_r101_8xb2-8e_youtubevis2021_20220823_092747-b7a7d7cc.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/mask2former/mask2former_r101_8xb2-8e_youtubevis2021_20220823_092747.json)                    |
| Mask2Former(200 queries) |  Swin-L  | pytorch |   8e    |   18.5   |       -        | 52.3 | [config](mask2former_swin-l-p4-w12-384-in21k_8xb2-8e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/mask2former/mask2former_swin-l-p4-w12-384-in21k_8xb2-8e_youtubevis2021_20220907_124752-c04b720e.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/mask2former/mask2former_swin-l-p4-w12-384-in21k_8xb2-8e_youtubevis2021_20220907_124752.json) |

## Get started

### 1. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. You can use the following command to start the training.

```shell
# Training Mask2Former on YouTube-VIS-2019 dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_train.sh \
    configs/vis/mask2former/mask2former_r50_8xb2-8e_youtubevis2019.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 2. Testing and evaluation

If you want to get the results of the [YouTube-VOS](https://youtube-vos.org/dataset/vis/) val/test set, please use the following command to generate result files that can be used for submission. It will be stored in `./youtube_vis_results.submission_file.zip`, you can modify the saved path in `test_evaluator` of the config.

```shell
# The number after config file represents the number of GPUs used.
./tools/dist_test.sh \
    configs/vis/mask2former/mask2former_r50_8xb2-8e_youtubevis2019.py 8 \
    --checkpoint ./checkpoints/xxx
```

If you want to know about more detailed usage of `test.py/dist_test.sh/slurm_test.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/demo_mot_vis.py \
    configs/vis/mask2former/mask2former_r50_8xb2-8e_youtubevis2019.py \
    --checkpoint ./checkpoints/xxx \
    --input demo/demo.mp4 \
    --output vis.mp4
```

If you want to know about more detailed usage of `demo_mot_vis.py`, please refer to this [document](../../../docs/en/user_guides/3_inference.md).
