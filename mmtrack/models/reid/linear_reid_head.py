import warnings

import torch.nn as nn
from mmcls.models.builder import HEADS
from mmcls.models.heads.base_head import BaseHead
from mmcls.models.losses import Accuracy
from mmcv.cnn import constant_init, normal_init
from mmdet.models.builder import build_loss

from .fc_module import FcModule


@HEADS.register_module()
class LinearReIDHead(BaseHead):
    """Linear head for re-identification.

    Args:
        num_fcs (int): Number of fcs.
        in_channels (int): Number of channels in the input.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        loss (dict, optional): Cross entropy loss to train the
            re-identificaiton module.
        loss_pairwise (dict, optional): Triplet loss to train the
            re-identificaiton module.
        topk (int, optional): Calculate topk accuracy. Default to False.
    """

    def __init__(self,
                 num_fcs,
                 in_channels,
                 fc_channels,
                 out_channels,
                 norm_cfg=None,
                 act_cfg=None,
                 num_classes=None,
                 loss=None,
                 loss_pairwise=None,
                 topk=(1, )):
        super(LinearReIDHead, self).__init__()
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        if not loss:
            if isinstance(num_classes, int):
                warnings.warn('Since cross entropy is not set, '
                              'the num_classes will be ignored.')
            if not loss_pairwise:
                raise ValueError('Please choose at least one loss in '
                                 'triplet loss and cross entropy loss.')
        elif not isinstance(num_classes, int):
            raise TypeError('The num_classes must be a current number, '
                            'if there is cross entropy loss.')
        self.loss_cls = build_loss(loss) if loss else None
        self.loss_triplet = build_loss(
            loss_pairwise) if loss_pairwise else None

        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.accuracy = Accuracy(topk=self.topk)

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(in_channels, self.fc_channels, self.norm_cfg,
                         self.act_cfg))
        in_channels = self.in_channels if self.num_fcs == 0 else \
            self.fc_channels
        self.fc_out = nn.Linear(in_channels, self.out_channels)
        if self.num_classes:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)

    def init_weights(self):
        """Initalize model weights."""
        normal_init(self.fc_out, mean=0, std=0.01, bias=0)
        if self.num_classes:
            constant_init(self.bn, 1, bias=0)
            normal_init(self.classifier, mean=0, std=0.01, bias=0)

    def simple_test(self, x):
        """Test without augmentation."""
        for m in self.fcs:
            x = m(x)
        feats = self.fc_out(x)
        return feats

    def forward_train(self, x, gt_label):
        """Model forward."""
        for m in self.fcs:
            x = m(x)
        feats = self.fc_out(x)
        losses = dict()
        if self.loss_triplet:
            losses['triplet_loss'] = self.loss_triplet(feats, gt_label)
        if self.loss_cls:
            feats_bn = self.bn(feats)
            out = self.classifier(feats_bn)
            losses['ce_loss'] = self.loss_cls(out, gt_label)
            # compute accuracy
            acc = self.accuracy(out, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        return losses
