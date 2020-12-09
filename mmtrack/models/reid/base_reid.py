from mmcls.models import ImageClassifier

from ..builder import REID


@REID.register_module()
class BaseReID(ImageClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError()
