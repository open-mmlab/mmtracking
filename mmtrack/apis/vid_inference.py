import warnings

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose

from mmtrack.models import build_model


def init_vid_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a video object detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Defaults to None.
        device (str, optional): The device where the model is put on. Defaults
            to 'cuda:0'
        cfg_options (dict, optional): Options to override some settings in the
            used config. Defaults to None.

    Returns:
        nn.Module: The constructed video object detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrains = None
    config.model.detector.pretrained = None
    model = build_model(config.model)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use ImageNet VID classes by default.')
            model.CLASSES = get_classes('imagenet_vid')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def vid_inference(model,
                  image,
                  frame_id,
                  ref_img_sampler=dict(frame_stride=10, num_left_ref_imgs=10)):
    """Inference image with the video object detector.

    Args:
        model (nn.Module): The loaded detector.
        image (ndarray): Loaded images.
        frame_id (int): Frame id.
        ref_img_sampler (dict): The configuration for sampling reference
            images. Only used under video detector of fgfa style. Defaults to
            dict(frame_stride=2, num_left_ref_imgs=10).

    Returns:
        dict[str : ndarray]: The detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if cfg.data.test.pipeline[0].type == 'LoadImageFromFile':
        data = dict(
            img=image.astype(np.float32).copy(),
            img_info=dict(frame_id=frame_id))

        # remove the "LoadImageFromFile" in pipeline
        test_pipeline = Compose(cfg.data.test.pipeline[1:])

    elif cfg.data.test.pipeline[0].type == 'LoadMultiImagesFromFile':
        data = [
            dict(
                img=image.astype(np.float32).copy(),
                img_info=dict(frame_id=frame_id))
        ]

        num_left_ref_imgs = ref_img_sampler.get('num_left_ref_imgs')
        frame_stride = ref_img_sampler.get('frame_stride')
        if frame_id == 0:
            for i in range(num_left_ref_imgs):
                one_ref_img = dict(
                    img=image.astype(np.float32).copy(),
                    img_info=dict(frame_id=frame_id))
                data.append(one_ref_img)
        elif frame_id % frame_stride == 0:
            one_ref_img = dict(
                img=image.astype(np.float32).copy(),
                img_info=dict(frame_id=frame_id))
            data.append(one_ref_img)

        # remove the "LoadMultiImagesFromFile" in pipeline
        test_pipeline = Compose(cfg.data.test.pipeline[1:])

    else:
        print('Not supported loading data pipeline type: '
              f'{cfg.data.test.pipeline[0].type}')
        raise NotImplementedError

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result
