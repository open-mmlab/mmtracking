# VIS (Video Instance Segmentation)

## Introduction

The goal of VIS task is simultaneous detection, segmentation and tracking of instances in videos. Here we implements `IDOL` based on contrastive learning and `VITA` series algorithms based on `Mask2Former`. Currently it provides advanced online and offline video instance segmentation algorithms. With a commitment to advancing the field of video instance segmentation, we will continually refine and enhance our framework to ensure it is both unified and efficient, providing the necessary nourishment for growth and development in this area.

In recent years, the online methods for video instance segmentation have witnessed significant advancements, largely attributed to the improvements in image-level object detection algorithms. Meanwhile, semi-online and offline paradigms are tapping into the vast potential offered by the temporal context in multiple frames, offering a more comprehensive approach to video analysis.

## Requirements

At the outset of this project, the dependencies used were as follows. Of course, this does not mean that you must strictly use the libraries and algorithms dependencies with the following versions. This is just a recommendation to make your use easier.

```
mmcv==2.0.0rc4
mmdet==3.0.0rc4
mmengine==0.4.0
```

## Citation

```BibTeX
@inproceedings{IDOL,
  title={In Defense of Online Models for Video Instance Segmentation},
  author={Wu, Junfeng and Liu, Qihao and Jiang, Yi and Bai, Song and Yuille, Alan and Bai, Xiang},
  booktitle={ECCV},
  year={2022},
}

@inproceedings{GenVIS,
  title={A Generalized Framework for Video Instance Segmentation},
  author={Heo, Miran and Hwang, Sukjun and Hyun, Jeongseok and Kim, Hanjung and Oh, Seoung Wug and Lee, Joon-Young and Kim, Seon Joo},
  booktitle={arXiv preprint arXiv:2211.08834},
  year={2022}
}

@inproceedings{VITA,
  title={VITA: Video Instance Segmentation via Object Token Association},
  author={Heo, Miran and Hwang, Sukjun and Oh, Seoung Wug and Lee, Joon-Young and Kim, Seon Joo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
