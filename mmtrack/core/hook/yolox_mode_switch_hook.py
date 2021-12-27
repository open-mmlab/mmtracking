# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS
from mmdet.core import YOLOXModeSwitchHook as _YOLOXModeSwitchHook


@HOOKS.register_module(force=True)
class YOLOXModeSwitchHook(_YOLOXModeSwitchHook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    The difference between this class and the class in mmdet is that the
    class in mmdet use `model.bbox_head.use_l1=True` to switch mode, while
    this class will check whether there is a detector module in the model
    firstly, then use `model.detector.bbox_head.use_l1=True` or
    `model.bbox_head.use_l1=True` to switch mode.
    """

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('No mosaic and mixup aug now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            runner.logger.info('Add additional L1 loss now!')
            if hasattr(model, 'detector'):
                model.detector.bbox_head.use_l1 = True
            else:
                model.bbox_head.use_l1 = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
