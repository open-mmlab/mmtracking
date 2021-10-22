# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmtrack.utils import get_root_logger


class BaseSingleObjectTracker(BaseModule, metaclass=ABCMeta):
    """Base class for single object tracker.

    Args:
        init_cfg (dict or list[dict]): Initialization config dict.
    """

    def __init__(self, init_cfg):
        super(BaseSingleObjectTracker, self).__init__(init_cfg)
        self.logger = get_root_logger()
        self.fp16_enabled = False

    def freeze_module(self, module):
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
    def forward_train(self, imgs, img_metas, search_img, search_img_metas,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            search_img (Tensor): of shape (N, 1, C, H, W) encoding input search
                images. 1 denotes there is only one search image for each
                exemplar image. Typically H and W equal to 255.

            search_img_metas (list[list[dict]]): The second list only has one
                element. The first list contains search image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        elif not isinstance(imgs, list):
            raise TypeError(
                f'imgs must be a list or tensor, but got {type(imgs)}')

        assert isinstance(img_metas, list)
        if isinstance(img_metas[0], dict):
            img_metas = [img_metas]
        elif not isinstance(img_metas[0], list):
            raise TypeError(
                'img_metas must be a List[List[dict]] or List[dict]')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', 'search_img'))
    def forward(self,
                img,
                img_metas,
                search_img=None,
                search_img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(
                img,
                img_metas,
                search_img=search_img,
                search_img_metas=search_img_metas,
                **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
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
