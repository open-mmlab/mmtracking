from mmcls.models import ImageClassifier

from ..builder import REID


@REID.register_module()
class BaseReID(ImageClassifier):
    """Base class for re-identification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_train(self, img, gt_label, **kwargs):
        """"Training forward function."""
        if img.ndim == 5:
            # change the shape of image tensor from NxSxCxHxW to NSxCxHxW
            # where S is the number of samples by triplet sampling
            img = img.view(-1, *img.shape[2:])
            # change the shape of label tensor from NxS to NS
            gt_label = gt_label.view(-1)
        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return loss

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        if img.nelement() > 0:
            x = self.extract_feat(img)
            return self.head.simple_test(x)
        else:
            return img.new_zeros(0, self.head.out_channels)
