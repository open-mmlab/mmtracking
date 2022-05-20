# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmtrack.core import TrackDataSample
from mmtrack.core.utils.misc import stack_batch


class BaseVideoDetector(BaseModule, metaclass=ABCMeta):
    """Base class for video object detector.

    Args:
        preprocess_cfg (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_value``,
            ``mean`` and ``std``. Default to None.
        init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self,
                 preprocess_cfg: dict = None,
                 init_cfg: dict = None) -> None:
        super(BaseVideoDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.preprocess_cfg = copy.deepcopy(preprocess_cfg)

        self.set_preprocess_cfg()

    def set_preprocess_cfg(self) -> None:
        """Set the preprocessing config for processing the input data."""
        if self.preprocess_cfg is not None:
            assert isinstance(self.preprocess_cfg, dict)
            preprocess_cfg = self.preprocess_cfg

            self.to_rgb = preprocess_cfg.get('to_rgb', False)
            self.pad_size_divisor = preprocess_cfg.get('pad_size_divisor', 0)
            self.pad_value = preprocess_cfg.get('pad_value', 0)
            self.register_buffer(
                'pixel_mean',
                torch.tensor(preprocess_cfg['mean']).view(1, -1, 1, 1), False)
            self.register_buffer(
                'pixel_std',
                torch.tensor(preprocess_cfg['std']).view(1, -1, 1, 1), False)
        else:
            # Only used to provide device information
            self.register_buffer('pixel_mean', torch.tensor(1), False)

        self.pad_size_divisor = 0
        self.pad_value = 0

    @property
    def device(self) -> torch.device:
        """Get the deive of model."""
        return self.pixel_mean.device

    def freeze_module(self, module: Union[List[str], Tuple[str], str]) -> None:
        """Freeze module during training."""
        if isinstance(module, str):
            modules = [module]
        else:
            if not (isinstance(module, list) or isinstance(module, tuple)):
                raise TypeError('module must be a str or a list.')
            else:
                modules = module
        for module in modules:
            m = getattr(self, module)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @property
    def with_detector(self) -> bool:
        """bool: whether the framework has a detector"""
        return hasattr(self, 'detector') and self.detector is not None

    @property
    def with_motion(self) -> bool:
        """bool: whether the framework has a motion model"""
        return hasattr(self, 'motion') and self.motion is not None

    @property
    def with_aggregator(self) -> bool:
        """bool: whether the framework has a aggregator"""
        return hasattr(self, 'aggregator') and self.aggregator is not None

    @abstractmethod
    def forward_train(self, batch_inputs: dict,
                      batch_data_samples: List[TrackDataSample],
                      **kwargs) -> dict:
        """
        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.

            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Return:
            dict: It should contain at least 3 keys: ``loss``,
            ``log_vars``, ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                    weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                    logger.
                - ``num_samples`` indicates the batch size (when the model
                    is DDP, it means the batch size on each GPU), which is
                    used for averaging the logs.
        """
        pass

    @abstractmethod
    def simple_test(self, batch_inputs: dict,
                    batch_data_samples: List[TrackDataSample],
                    **kwargs) -> List[TrackDataSample]:
        """
        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor, Optional): The reference images.

            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            list(obj:`TrackDataSample`): Tracking results of the
            input images. Each TrackDataSample usually contains
            ``pred_det_instances`` or ``pred_track_instances``.
        """
        pass

    @auto_fp16(apply_to=('batch_inputs', ))
    def _forward_train(self, batch_inputs, batch_data_samples, **kwargs):
        """Only for support fp16 training."""
        return self.forward_train(batch_inputs, batch_data_samples, **kwargs)

    @auto_fp16(apply_to=('batch_inputs', ))
    def _forward_test(self, batch_inputs, batch_data_samples, **kwargs):
        """Only for support fp16 testing."""
        return self.simple_test(
            batch_inputs, batch_data_samples, rescale=True, **kwargs)

    def forward(self,
                data: List[dict],
                optimizer: Optional[Union[torch.optim.Optimizer, dict]] = None,
                return_loss: bool = False,
                **kwargs) -> Union[dict, List[TrackDataSample]]:
        """The iteration step during training and testing. This method defines
        an iteration step during training and testing, except for the back
        propagation and optimizer updating during training, which are done in
        an optimizer hook.

        Args:
            data (list[dict]): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer`, dict, Optional): The
                optimizer of runner. This argument is unused and reserved.
                Default to None.
            return_loss (bool): Whether to return loss. In general,
                it will be set to True during training and False
                during testing. Default to False.

        Returns:
            during training
                dict: It should contain at least 3 keys: ``loss``,
                ``log_vars``, ``num_samples``.
                    - ``loss`` is a tensor for back propagation, which can be a
                      weighted sum of multiple losses.
                    - ``log_vars`` contains all the variables to be sent to the
                        logger.
                    - ``num_samples`` indicates the batch size (when the model
                        is DDP, it means the batch size on each GPU), which is
                        used for averaging the logs.

            during testing
                list(obj:`TrackDataSample`): Tracking results of the
                input images. Each TrackDataSample usually contains
                ``pred_det_instances`` or ``pred_track_instances``.
        """
        batch_inputs, batch_data_samples = self.preprocss_data(data)

        if return_loss:
            losses = self._forward_train(batch_inputs, batch_data_samples,
                                         **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(batch_data_samples))
            return outputs
        else:
            assert len(data) == 1, \
                'Only support simple test with batch_size==1 currently. ' \
                'Aug-test and batch_size > 1 are not supported yet'
            return self._forward_test(batch_inputs, batch_data_samples,
                                      **kwargs)

    def preprocss_data(self, data: List[dict]) -> Tuple:
        """ Process input data during training and simple testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 item.
                 - batch_inputs (dict[Tensor]): The batch input tensor.
                 - batch_data_samples (list[:obj:`TrackDataSample`]): The Data
                     Samples. It usually includes information such as
                     `gt_instance`.
        """
        batch_data_samples = [
            data_['data_sample'].to(self.device) for data_ in data
        ]

        # Collate inputs (list of dict to dict of list)
        inputs = {
            key: [_data['inputs'][key].to(self.device) for _data in data]
            for key in data[0]['inputs']
        }

        batch_inputs = dict()
        for imgs_key, imgs in inputs.items():
            if self.preprocess_cfg is None:
                # YOLOX does not need preprocess_cfg
                batch_inputs[imgs_key] = stack_batch(imgs).float()
            else:
                # imgs is a list contain multiple Tensor of imgs.
                # The shape of imgs[0] is (T, C, H, W).
                channel = imgs[0].size(1)
                if self.to_rgb and channel == 3:
                    imgs = [_img[:, [2, 1, 0], ...] for _img in imgs]
                imgs = [(_img - self.pixel_mean) / self.pixel_std
                        for _img in imgs]
                batch_inputs[imgs_key] = stack_batch(imgs,
                                                     self.pad_size_divisor,
                                                     self.pad_value)
        return batch_inputs, batch_data_samples

    def _parse_losses(self, losses: dict) -> Tuple:
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img` det_bboxes or
                (det_bboxes, det_masks). The value of key 'det_bboxes'
                is list with length num_classes, and each element in list
                is ndarray with shape(n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        # TODO: make it support tracking
        img = mmcv.imread(img)
        img = img.copy()
        assert isinstance(result, dict)
        bbox_results = result.get('det_bboxes', None)
        mask_results = result.get('det_masks', None)
        if isinstance(mask_results, tuple):
            mask_results = mask_results[0]  # ms rcnn
        bboxes = np.vstack(bbox_results)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_results)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if mask_results is not None and len(labels) > 0:  # non empty
            masks = mmcv.concat_list(mask_results)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = masks[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
