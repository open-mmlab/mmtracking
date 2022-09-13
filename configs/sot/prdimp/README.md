# Probabilistic Regression for Visual Tracking

## Abstract

<!-- [ABSTRACT] -->

Visual tracking is fundamentally the problem of regressing the state of the target in each video frame. While significant progress has been achieved, trackers are still prone to failures and inaccuracies. It is therefore crucial to represent the uncertainty in the target estimation. Although current prominent paradigms rely on estimating a state-dependent confidence score, this value lacks a clear probabilistic interpretation, complicating its use.
In this work, we therefore propose a probabilistic regression formulation and apply it to tracking. Our network predicts the conditional probability density of the target state given an input image. Crucially, our formulation is capable of modeling label noise stemming from inaccurate annotations and ambiguities in the task. The regression network is trained by minimizing the Kullback-Leibler divergence. When applied for tracking, our formulation not only allows a probabilistic representation of the output, but also substantially improves the performance. Our tracker sets a new state-of-the-art on six datasets, achieving 59.8% AUC on LaSOT and 75.8% Success on TrackingNet. The code and models are available at [this https URL](https://github.com/visionml/pytracking).

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/188844862-9bec1963-54f4-4c1c-b013-52fec3811465.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{Danelljan2020Probabilistic,
  title={Probabilistic Regression for Visual Tracking},
  author={Danelljan, Martin and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2020}
}
```

## Results and models

### LaSOT

We provide the last-epoch model with its configuration and training log.

| Method | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                                   Config                                   |                                                                                                                                                  Download                                                                                                                                                  |
| :----: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PrDiMP |   R-50   |   -   |   50e   |   13.9   |       -        |  59.7   |      67.7      |   60.5    | [config](prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200-b7dbeca4.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200.json) |

### TrackingNet

The last-epoch model on LaSOT is submitted to [the evaluation server on TrackingNet Challenge](https://eval.ai/web/challenges/challenge-page/1805/). We provide the model with its configuration and training log.

| Method | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                                      Config                                      |                                                                                                                                                  Download                                                                                                                                                  |
| :----: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PrDiMP |   R-50   |   -   |   50e   |   13.9   |       -        |  75.2   |      80.5      |   70.1    | [config](prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-trackingnet.py) | [model](https://download.openmmlab.com/mmtracking/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200-b7dbeca4.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200.json) |

### GOT10k

The results of PrDiMP in GOT10k are reimplemented by ourselves. We only use the GOT10k train set to train the model, which is the common setting on GOT10k, while the setting on the PrDiMP paper is using the GOT10k, LaSOT, TrackingNet and coco to train the model. The result on our setting is lower about 1 than the original PrDiMP setting. The last-epoch model is submitted to [the evaluation server on GOT10k Challenge](http://got-10k.aitestunion.com/). We provide the model with its configuration and training log.

| Method | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) |  AO  | SR<sub>0.5</sub> | SR<sub>0.75</sub> |                  Config                  |                                                                                                                Download                                                                                                                |
| :----: | :------: | :---: | :-----: | :------: | :------------: | :--: | :--------------: | :---------------: | :--------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PrDiMP |   R-50   |   -   |   50e   |   13.9   |       -        | 62.9 |       72.5       |       52.8        | [config](prdimp_r50_8xb10-50e_got10k.py) | [model](https://download.openmmlab.com/mmtracking/sot/prdimp/prdimp_r50_8xb10-50e_got10k_20220907_173919-fa24df25.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/prdimp/prdimp_r50_8xb10-50e_got10k_20220907_173919.json) |

## Get started

### 1. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. The following is an example of training PrDiMP tested on LaSOT dataset. The model on GOT10k is similar like this.

```shell
# Training PrDiMP on LaSOT、TrackingNet and coco dataset.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs
./tools/dist_train.sh \
    configs/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 2. Testing and evaluation

**2.1 Example on LaSOT dataset**

```shell
# Test PrDiMP on LaSOT testset
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot.py 8 \
    --checkpoint ./checkpoints/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200-b7dbeca4.pth
```

**2.1 Example on TrackingNet and GOT10k datasets**

If you want to get the results of the [TrackingNet](https://eval.ai/web/challenges/challenge-page/1805/) and [GOT10k](http://got-10k.aitestunion.com/), please use the following commands to generate result files that can be used for submission. You can modify the saved path in `test_evaluator` of the config.

```shell
# Test PrDIMP on TrackingNet testset.
# The result is stored in `./results/prdimp_trackingnet.zip` by default.
# We use the lasot checkpoint on LaSOT to test on the TrackingNet.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-trackingnet.py 8 \
    --checkpoint ./checkpoints/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200-b7dbeca4.pth
```

```shell
# Test PrDiMP on GOT10k testset.
# The result is stored in `./results/prdimp_got10k.zip` by default.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/sot/prdimp/prdimp_r50_8xb10-50e_got10k.py 8 \
    --checkpoint ./checkpoints/prdimp_r50_8xb10-50e_got10k_20220907_173919-fa24df25.pth
```

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/demo_sot.py \
    configs/sot/prdimp/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot.py \
    --checkpoint ./checkpoints/prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_test-lasot_20220822_082200-b7dbeca4.pth \
    --input demo/demo.mp4 \
    --output sot.mp4
```

If you want to know about more detailed usage of `demo_sot.py`, please refer to this [document](../../../docs/en/user_guides/3_inference.md).
