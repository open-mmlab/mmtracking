# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import tempfile
import warnings
from typing import Optional, Union

import mmengine
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmengine.dataset import Compose
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint
from torch import nn

from mmtrack.registry import MODELS
from mmtrack.utils import SampleList


def init_model(config: Union[str, mmengine.Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None,
               verbose_init_params: bool = False) -> nn.Module:
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (Optional[str], optional): Checkpoint path. Defaults to
            None.
        device (str, optional): The device that the model inferences on.
            Defaults to `cuda:0`.
        cfg_options (Optional[dict], optional): Options to override some
            settings in the used config. Defaults to None.
        verbose_init_params (bool, optional): Whether to print the information
            of initialized parameters to the console. Defaults to False.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    model = MODELS.build(config.model)

    if not verbose_init_params:
        # Creating a temporary file to record the information of initialized
        # parameters. If not, the information of initialized parameters will be
        # printed to the console because of the call of
        # `mmcv.runner.BaseModule.init_weights`.
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        file_handler = logging.FileHandler(tmp_file.name, mode='w')
        logger = MMLogger.get_current_instance()
        logger.addHandler(file_handler)
        # We need call `init_weights()` to load pretained weights in MOT
        # task.
        model.init_weights()
        file_handler.close()
        logger.removeHandler(file_handler)
        tmp_file.close()
        os.remove(tmp_file.name)
    else:
        # We need call `init_weights()` to load pretained weights in MOT task.
        model.init_weights()

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmtrack 1.x
            model.dataset_meta = checkpoint_meta['dataset_meta']
        elif 'classes' in checkpoint_meta:
            # < mmtrack 1.x
            classes = checkpoint_meta['classes']
            model.dataset_meta = {'classes': classes}

    # Some methods don't load checkpoints or checkpoints don't contain
    # 'dataset_meta'
    if not hasattr(model, 'dataset_meta'):
        warnings.simplefilter('once')
        warnings.warn('dataset_meta or class names are missed, '
                      'use None by default.')
        model.dataset_meta = {'classes': None}

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_mot(model: nn.Module, img: np.ndarray,
                  frame_id: int) -> SampleList:
    """Inference image(s) with the mot model.

    Args:
        model (nn.Module): The loaded mot model.
        img (np.ndarray): Loaded image.
        frame_id (int): frame id.

    Returns:
        SampleList: The tracking data samples.
    """
    cfg = model.cfg
    data = dict(
        img=img.astype(np.float32), frame_id=frame_id, ori_shape=img.shape[:2])
    # remove the "LoadImageFromFile" and "LoadTrackAnnotations" in pipeline
    test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline[2:])
    data = test_pipeline(data)

    if not next(model.parameters()).is_cuda:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        data = mmengine.dataset.default_collate([data])
        result = model.test_step(data)[0]
    return result


def inference_sot(model: nn.Module, image: np.ndarray, init_bbox: np.ndarray,
                  frame_id: int) -> SampleList:
    """Inference image with the single object tracker.

    Args:
        model (nn.Module): The loaded tracker.
        image (np.ndarray): Loaded images.
        init_bbox (np.ndarray): The target needs to be tracked.
        frame_id (int): frame id.

    Returns:
        SampleList: The tracking data samples.
    """
    cfg = model.cfg
    data = dict(
        img=image.astype(np.float32),
        gt_bboxes=np.array(init_bbox).astype(np.float32),
        frame_id=frame_id,
        ori_shape=image.shape[:2])
    # remove the "LoadImageFromFile" and "LoadAnnotations" in pipeline
    test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline[2:])
    data = test_pipeline(data)

    if not next(model.parameters()).is_cuda:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        data = mmengine.dataset.default_collate([data])
        result = model.test_step(data)[0]
    return result


def inference_vid(
    model: nn.Module,
    image: np.ndarray,
    frame_id: int,
    ref_img_sampler: dict = dict(frame_stride=2, num_left_ref_imgs=10)
) -> SampleList:
    """Inference image with the video object detector.

    Args:
        model (nn.Module): The loaded detector.
        image (ndarray): Loaded images.
        frame_id (int): Frame id.
        ref_img_sampler (dict): The configuration for sampling reference
            images. Only used under video detector of fgfa style. Defaults to
            dict(frame_stride=2, num_left_ref_imgs=10).

    Returns:
        SampleList: The detection results.
    """
    cfg = model.cfg

    first_transform = cfg.test_dataloader.dataset.pipeline[0]
    if first_transform.type == 'LoadImageFromFile':
        data = dict(
            img=image.astype(np.float32).copy(),
            frame_id=frame_id,
            ori_shape=image.shape[:2])
        # remove the "LoadImageFromFile" in pipeline
        test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline[1:])
    elif first_transform.type == 'TransformBroadcaster':
        assert first_transform.transforms[0].type == 'LoadImageFromFile'
        # Only used under video detector of fgfa style.
        data = dict(
            img=[image.astype(np.float32).copy()],
            frame_id=[frame_id],
            ori_shape=[image.shape[:2]])

        num_left_ref_imgs = ref_img_sampler.get('num_left_ref_imgs')
        frame_stride = ref_img_sampler.get('frame_stride')
        if frame_id == 0:
            for i in range(num_left_ref_imgs):
                data['img'].append(image.astype(np.float32).copy())
                data['frame_id'].append(frame_id)
                data['ori_shape'].append(image.shape[:2])
        elif frame_id % frame_stride == 0:
            data['img'].append(image.astype(np.float32).copy())
            data['frame_id'].append(frame_id)
            data['ori_shape'].append(image.shape[:2])
        # In order to pop the LoadImageFromFile, test_pipeline[0] is
        # `TransformBroadcaster` and test_pipeline[0].transforms[0]
        # is 'LoadImageFromFile'.
        test_pipeline = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
        test_pipeline[0].transforms.pop(0)
        test_pipeline = Compose(test_pipeline)
    else:
        print('Not supported loading data pipeline type: '
              f'{first_transform.type}')
        raise NotImplementedError

    data = test_pipeline(data)

    if not next(model.parameters()).is_cuda:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        data = mmengine.dataset.default_collate([data])
        result = model.test_step(data)[0]
    return result
