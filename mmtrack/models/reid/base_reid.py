from mmcls.models import ImageClassifier

from ..builder import REID


@REID.register_module()
class BaseReID(ImageClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError()

    def simple_test(self, img):
        """Test without augmentation."""
        if img.nelement() > 0:
            x = self.extract_feat(img)
            return self.head.simple_test(x)
        else:
            return img.new_zeros(0, self.head.out_channels)
