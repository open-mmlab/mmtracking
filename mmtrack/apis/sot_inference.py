import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose

from mmtrack.models import build_model


def init_sot_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a single object tracker from config file.

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
        nn.Module: The constructed single object tracker.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrains = None
    model = build_model(config.model)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def sot_inference(model, image, init_bbox, frame_id):
    """Inference image with the single object tracker.

    Args:
        model (nn.Module): The loaded tracker.
        image (ndarray): Loaded images.
        init_bbox (ndarray): The target needs to be tracked.
        frame_id (int): frame id.

    Returns:
        dict[str : ndarray]: The tracking results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    data = dict(
        img=image.astype(np.float32),
        gt_bboxes=np.array(init_bbox).astype(np.float32),
        img_info=dict(frame_id=frame_id))
    # remove the "LoadImageFromFile" and "LoadAnnotations" in pipeline
    test_pipeline = Compose(cfg.data.test.pipeline[2:])
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
