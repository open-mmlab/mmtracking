# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models import ImageClassifier
from mmcv.runner import auto_fp16

from ..builder import REID


@REID.register_module()
class BaseReID(ImageClassifier):
    """Base class for re-identification."""

    def forward_train(self, img, gt_label, **kwargs):
        """"Training forward function."""
        if img.ndim == 5:
            # change the shape of image tensor from NxSxCxHxW to NSxCxHxW
            # where S is the number of samples by triplet sampling
            img = img.view(-1, *img.shape[2:])
            # change the shape of label tensor from NxS to NS
            gt_label = gt_label.view(-1)
        x = self.extract_feat(img)
        head_outputs = self.head.forward_train(x[0])

        losses = dict()
        reid_loss = self.head.loss(gt_label, *head_outputs)
        losses.update(reid_loss)
        return losses

    @auto_fp16(apply_to=('img', ), out_fp32=True)
    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        if img.nelement() > 0:
            x = self.extract_feat(img)
            head_outputs = self.head.forward_train(x[0])
            feats = head_outputs[0]
            return feats
        else:
            return img.new_zeros(0, self.head.out_channels)
