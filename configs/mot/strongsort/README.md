# StrongSORT: Make DeepSORT Great Again

## Abstract

<!-- [ABSTRACT] -->

Existing Multi-Object Tracking (MOT) methods can be roughly classified as tracking-by-detection and joint-detection-association paradigms. Although the latter has elicited more attention and demonstrates comparable performance relative to the former, we claim that the tracking-by-detection paradigm is still the optimal solution in terms of tracking accuracy. In this paper, we revisit the classic tracker DeepSORT and upgrade it from various aspects, i.e., detection, embedding and association. The resulting tracker, called StrongSORT, sets new HOTA and IDF1 records on MOT17 and MOT20. We also present two lightweight and plug-and-play algorithms to further refine the tracking results. Firstly, an appearance-free link model (AFLink) is proposed to associate short tracklets into complete trajectories. To the best of our knowledge, this is the first global link model without appearance information. Secondly, we propose Gaussian-smoothed interpolation (GSI) to compensate for missing detections. Instead of ignoring motion information like linear interpolation, GSI is based on the Gaussian process regression algorithm and can achieve more accurate localizations. Moreover, AFLink and GSI can be plugged into various trackers with a negligible extra computational cost (591.9 and 140.9 Hz, respectively, on MOT17). By integrating StrongSORT with the two algorithms, the final tracker StrongSORT++ ranks first on MOT17 and MOT20 in terms of HOTA and IDF1 metrics and surpasses the second-place one by 1.3 - 2.2. Code will be released soon.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/99722489/185282811-ec82bdf6-8889-4f01-9c4d-a8e104f775b7.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@article{du2022strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Song, Yang and Yang, Bo and Zhao, Yanyun},
  journal={arXiv preprint arXiv:2202.13514},
  year={2022}
}
```

## Results and models on MOT17

|    Method    | Detector | ReID |           Train Set           |    Test Set    | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                        Config                                        |                                                                                                                                                                                   Download                                                                                                                                                                                    |
| :----------: | :------: | :--: | :---------------------------: | :------------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| StrongSORT++ | YOLOX-X  | R50  | CrowdHuman + MOT17-half-train | MOT17-half-val |   N    |       -        | 70.9 | 78.3 | 83.2 | 15336 | 19065 |  621  | [config](strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot17-private-half_20220812_192036-b6c9ce9a.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) [AFLink](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth) |

## Results and models on MOT20

|    Method    | Detector | ReID |        Train Set         |  Test Set  | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                    Config                                     |                                                                                                                                                                                         Download                                                                                                                                                                                         |
| :----------: | :------: | :--: | :----------------------: | :--------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :---------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| StrongSORT++ | YOLOX-X  | R50  | CrowdHuman + MOT20-train | MOT20-test |   N    |       -        | 62.9 | 75.5 | 77.3 | 29043 | 96155 | 1640  | [config](strongsort_yolox_x_8xb4-80e_crowdhuman-mot20train_test-mot20test.py) | [detector](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot20-private_20220812_192123-77c014de.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth) [AFLink](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth) |
