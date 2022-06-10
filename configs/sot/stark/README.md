# Learning Spatio-Temporal Transformer for Visual Tracking

## Abstract

<!-- [ABSTRACT] -->

In this paper, we present a new tracking architecture with an encoder-decoder transformer as the key component. The encoder models the global spatio-temporal feature dependencies between target objects and search regions, while the decoder learns a query embedding to predict the spatial positions of the target objects. Our method casts object tracking as a direct bounding box prediction problem, without using any proposals or predefined anchors. With the encoder-decoder transformer, the prediction of objects just uses a simple fully-convolutional network, which estimates the corners of objects directly. The whole method is end-to-end, does not need any postprocessing steps such as cosine window and bounding box smoothing, thus largely simplifying existing tracking pipelines. The proposed tracker achieves state-of-the-art performance on five challenging short-term and long-term benchmarks, while running at real-time speed, being 6× faster than Siam R-CNN. Code and models are open-sourced at [here](https://github.com/researchmm/Stark).

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/155925284-38187ef3-30f2-434f-bed8-133c0061f3e3.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{yan2021learning,
  title={Learning spatio-temporal transformer for visual tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10448--10457},
  year={2021}
}
```

## Results and models

The STARK is trained in 2 stages. We denote the 1st-stage model as `STARK-ST1`, and denote the 2nd-stage model as `STARK-ST2`. The following models we provide are the last-epoch models by default.

Models from the 2 stages have different configurations. For example, `stark_st1_r50_500e_got10k` is the configuration of the 1st-stage model and `stark_st2_r50_50e_got10k` is the configuration of the 2nd-stage model.

Note: We have to pass an extra parameter `cfg-options` containing the key `load_from` from shell command to load the pretrained 1st-stage model when training the 2nd-stage model. Here is an example:

```
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --cfg-options load_from=${STARK-ST1 model}
```

### LaSOT

We provide the last-epoch model with its configuration and training log.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                Config                 |                                                                                                                                       Download                                                                                                                                       |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :-----------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| STARK-ST1 |   R-50   |   -   |  500e   |   8.45   |       -        |  67.0   |      77.3      |   71.7    | [config](stark_st1_r50_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220414_185654-9c19e39e.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220414_185654.log.json) |
| STARK-ST2 |   R-50   |   -   |   50e   |   2.31   |       -        |  67.8   |      78.5      |   73.0    | [config](stark_st2_r50_50e_lasot.py)  |   [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201.log.json)   |

### TrackingNet

The results of STARK in TrackingNet are reimplemented by ourselves. The last-epoch model on TrackingNet is submitted to [the evaluation server on TrackingNet Challenge](http://eval.tracking-net.org/web/challenges/challenge-page/39/submission). We provide the model with its configuration and training log.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                Config                 |                                                                                                                                       Download                                                                                                                                       |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :-----------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| STARK-ST1 |   R-50   |   -   |  500e   |   8.45   |       -        |  80.3   |      85.0      |   77.7    | [config](stark_st1_r50_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220414_185654-9c19e39e.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220414_185654.log.json) |
| STARK-ST2 |   R-50   |   -   |   50e   |   2.31   |       -        |  81.4   |      86.2      |   79.0    | [config](stark_st2_r50_50e_lasot.py)  |   [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201.log.json)   |

### GOT10k

The results of STARK in GOT10k are reimplemented by ourselves. The last-epoch model on GOT10k is submitted to [the evaluation server on GOT10k Challenge](http://got-10k.aitestunion.com/). We provide the model with its configuration and training log.

|  Method   | Backbone | Style | Lr schd | Mem (GB) | Inf time (fps) | Success | Norm precision | Precision |                Config                 |                                                                                                                                         Download                                                                                                                                         |
| :-------: | :------: | :---: | :-----: | :------: | :------------: | :-----: | :------------: | :-------: | :-----------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| STARK-ST1 |   R-50   |   -   |  500e   |   8.45   |       -        |  68.1   |      77.4      |   62.4    | [config](stark_st1_r50_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_got10k/stark_st1_r50_500e_got10k_20220223_125400-40ead158.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_got10k/stark_st1_r50_500e_got10k_20220223_125400.log.json) |
| STARK-ST2 |   R-50   |   -   |   50e   |   2.31   |       -        |  68.3   |      77.6      |   62.7    | [config](stark_st2_r50_50e_lasot.py)  |   [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_got10k/stark_st2_r50_50e_got10k_20220226_124213-ee39bbff.pth) \| [log](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_got10k/stark_st2_r50_50e_got10k_20220226_124213.log.json)   |
