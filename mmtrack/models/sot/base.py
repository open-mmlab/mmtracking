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


class BaseSingleObjectTracker(BaseModule, metaclass=ABCMeta):
    """Base class for single object tracker.

    Args:
        preprocess_cfg (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_value``,
            ``mean`` and ``std``. Default to None.
        init_cfg (dict or list[dict]): Initialization config dict.
    """

    def __init__(self,
                 preprocess_cfg: dict = None,
                 init_cfg: dict = None) -> None:
        super(BaseSingleObjectTracker, self).__init__(init_cfg)
        self.preprocess_cfg = copy.deepcopy(preprocess_cfg)
        self.set_preprocess_cfg()

    def set_preprocess_cfg(self) -> None:
        """Set the preprocessing config for processing the input data."""
        self.pad_size_divisor = 0
        self.pad_value = 0
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
    def with_backbone(self):
        """bool: whether the framework has a backbone"""
        return hasattr(self, 'backbone') and self.backbone is not None

    @property
    def with_neck(self):
        """bool: whether the framework has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        """bool: whether the framework has a head"""
        return hasattr(self, 'head') and self.head is not None

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
            dict: A dictionary of loss components.
        """
        pass

    @abstractmethod
    def simple_test(self, batch_inputs: dict,
                    batch_data_samples: List[TrackDataSample],
                    **kwargs) -> List[TrackDataSample]:
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

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    color='green',
                    thickness=1,
                    show=False,
                    win_name='',
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str or ndarray): The image to be displayed.
            result (dict): Tracking result.
                The value of key 'track_bboxes' is ndarray with shape (5, )
                in [tl_x, tl_y, br_x, br_y, score] format.
            color (str or tuple or Color, optional): color of bbox.
                Defaults to green.
            thickness (int, optional): Thickness of lines.
                Defaults to 1.
            show (bool, optional): Whether to show the image.
                Defaults to False.
            win_name (str, optional): The window name.
                Defaults to ''.
            wait_time (int, optional): Value of waitKey param.
                Defaults to 0.
            out_file (str, optional): The filename to write the image.
                Defaults to None.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_bboxes = result.get('track_bboxes', None)
        assert track_bboxes.ndim == 1
        assert track_bboxes.shape[0] == 5

        track_bboxes = track_bboxes[:4]
        mmcv.imshow_bboxes(
            img,
            track_bboxes[np.newaxis, :],
            colors=color,
            thickness=thickness,
            show=show,
            win_name=win_name,
            wait_time=wait_time,
            out_file=out_file)
        return img
