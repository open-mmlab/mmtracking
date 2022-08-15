# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import DATASETS, build_dataset

from .base_sot_dataset import BaseSOTDataset
from .builder import build_dataloader
from .coco_video_dataset import CocoVideoDataset
from .dancetrack_dataset import DanceTrackDataset
from .dataset_wrappers import RandomSampleConcatDataset
from .got10k_dataset import GOT10kDataset
from .imagenet_vid_dataset import ImagenetVIDDataset
from .lasot_dataset import LaSOTDataset
from .mot_challenge_dataset import MOTChallengeDataset
from .otb_dataset import OTB100Dataset
from .parsers import CocoVID
from .pipelines import PIPELINES
from .reid_dataset import ReIDDataset
from .sot_coco_dataset import SOTCocoDataset
from .sot_imagenet_vid_dataset import SOTImageNetVIDDataset
from .sot_test_dataset import SOTTestDataset
from .sot_train_dataset import SOTTrainDataset
from .tao_dataset import TaoDataset
from .trackingnet_dataset import TrackingNetDataset
from .uav123_dataset import UAV123Dataset
from .vot_dataset import VOTDataset
from .youtube_vis_dataset import YouTubeVISDataset

__all__ = [
    'DATASETS', 'PIPELINES', 'build_dataloader', 'build_dataset', 'CocoVID',
    'CocoVideoDataset', 'ImagenetVIDDataset', 'MOTChallengeDataset',
    'ReIDDataset', 'SOTTrainDataset', 'SOTTestDataset', 'LaSOTDataset',
    'UAV123Dataset', 'TrackingNetDataset', 'OTB100Dataset',
    'YouTubeVISDataset', 'GOT10kDataset', 'VOTDataset', 'BaseSOTDataset',
    'SOTCocoDataset', 'SOTImageNetVIDDataset', 'RandomSampleConcatDataset',
    'TaoDataset', 'DanceTrackDataset'
]
